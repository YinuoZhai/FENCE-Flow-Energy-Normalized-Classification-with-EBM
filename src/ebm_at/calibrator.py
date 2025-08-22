import torch
import torch, torch.nn as nn
from ebm_at.utils import bn_eval
from ebm_at.attacks import _pgd_adversary

class RouterCalibrator:
    def __init__(
        self,
        calib_loader,
        interval: int = 5,              # Calibrator activation interval
        max_batches: int = 20,
        pgd_steps: int = 10,             # Calibrator PGD power
        clean_fpr_max: float = 0.05,         # upper bound of a clean being identified as adv
        nat_fpr_max: float = 0.10,          # upper bound of a clean being identified as nat
        router_logic: str = "AND",
        mask_alpha_nat: float = 0.0,
        q_grid = (0.60, 0.70, 0.80, 0.85, 0.90, 0.95),
        target_adv_frac: float = 0.70,
        wandb_prefix: str = "calib",
        use_wandb: bool = False,
    ):
        self.loader = calib_loader
        self.interval = int(interval)
        self.max_batches = int(max_batches)
        self.pgd_steps = int(pgd_steps)
        self.clean_fpr_max = float(clean_fpr_max)
        self.nat_fpr_max = float(nat_fpr_max)
        self.router_logic = str(router_logic)
        self.target_adv_frac = float(target_adv_frac)
        self.q_grid = tuple(q_grid)
        self.wandb_prefix = wandb_prefix
        self.use_wandb = use_wandb
        self.last = None

    @torch.no_grad()
    def _quantiles(self, vals: torch.Tensor):
        # vals: (N,)
        return [torch.quantile(vals, torch.tensor(q)).item() for q in self.q_grid]


    def _rates(self, Es: torch.Tensor, Gs: torch.Tensor, tauE: float, tauG: float):
        """return (adv_rate, nat_rate)，nat_rate only AND"""
        if self.router_logic.upper() == "AND":
            is_adv = (Es > tauE) & (Gs > tauG)
            is_nat = (Es > tauE) & ~(Gs > tauG)
        else:  # "OR"
            is_adv = (Es > tauE) | (Gs > tauG)
            is_nat = (Es > tauE) & ~(Gs > tauG)
        adv_rate = is_adv.float().mean().item()
        nat_rate = is_nat.float().mean().item()
        return adv_rate, nat_rate


    def _collect_distributions(self, model, cfg):
        Es_c_list, Gs_c_list, Es_a_list, Gs_a_list = [], [], [], []
        ce = nn.CrossEntropyLoss()

        with bn_eval(model), torch.enable_grad():
            for i, (x, y) in enumerate(self.loader):
                if i >= self.max_batches: break
                x = x.to(cfg.device); y = y.to(cfg.device)

                # ----- clean -----
                x_c = x.detach().requires_grad_(True)
                out_c = model(x_c)
                E_c  = model.combined_energy_from_out(out_c, y=y, alpha=cfg.alpha_energy_fusion,
                                     update_norm=False, cls_mode=cfg.cls_energy_mode, cls_tau=cfg.cls_tau)                         # (B,)
                g_c  = torch.autograd.grad(E_c.sum(), x_c, retain_graph=False, create_graph=False)[0]
                G_c  = g_c.flatten(1).norm(dim=1)           # (B,)
                Es_c_list.append(E_c.detach().reshape(-1).cpu())
                Gs_c_list.append(G_c.detach().reshape(-1).cpu())

                # ----- pgd -----
                x_adv = _pgd_adversary(model, x, y, cfg, steps=self.pgd_steps)
                x_a   = x_adv.detach().requires_grad_(True)
                out_a = model(x_a)
                E_a   = model.combined_energy_from_out(out_a, y=y, alpha=cfg.alpha_energy_fusion,
                                     update_norm=False, cls_mode=cfg.cls_energy_mode, cls_tau=cfg.cls_tau)                         # (B,)
                g_a   = torch.autograd.grad(E_a.sum(), x_a, retain_graph=False, create_graph=False)[0]
                G_a   = g_a.flatten(1).norm(dim=1)           # (B,)
                Es_a_list.append(E_a.detach().reshape(-1).cpu())
                Gs_a_list.append(G_a.detach().reshape(-1).cpu())


        if (len(Es_c_list) == 0) or (len(Es_a_list) == 0):
            raise RuntimeError("Calib loader produced no samples.")

        Es_c = torch.cat([t.reshape(-1) for t in Es_c_list], dim=0)
        Gs_c = torch.cat([t.reshape(-1) for t in Gs_c_list], dim=0)
        Es_a = torch.cat([t.reshape(-1) for t in Es_a_list], dim=0)
        Gs_a = torch.cat([t.reshape(-1) for t in Gs_a_list], dim=0)

        assert Es_c.numel() == Gs_c.numel(), f"Clean counts mismatch: {Es_c.numel()} vs {Gs_c.numel()}"
        assert Es_a.numel() == Gs_a.numel(), f"PGD counts mismatch: {Es_a.numel()} vs {Gs_a.numel()}"

        return Es_c, Gs_c, Es_a, Gs_a

    def calibrate(self, model, cfg):
        model.eval()
        Es_c, Gs_c, Es_a, Gs_a = self._collect_distributions(model, cfg)

        cand_E = sorted(set(self._quantiles(Es_c) + self._quantiles(Es_a)))
        cand_G = sorted(set(self._quantiles(Gs_c) + self._quantiles(Gs_a)))

        best = None
        for tauE in cand_E:
            for tauG in cand_G:
                clean_adv, clean_nat = self._rates(Es_c, Gs_c, tauE, tauG)
                # “clean→ADV” False Positive Rate constraint (FPR)
                if clean_adv > self.clean_fpr_max:
                    continue
                # “clean→NAT” FPR constraint
                if self.router_logic.upper() == "AND" and (clean_nat > self.nat_fpr_max):
                    continue

                pgd_adv, _ = self._rates(Es_a, Gs_a, tauE, tauG)

                # Within the feasible set, make pgd_adv close to target_adv_frac; and prefer pgd_adv to be larger.
                score = (abs(pgd_adv - self.target_adv_frac), -pgd_adv)
                if (best is None) or (score < best[0]):
                    best = (score, tauE, tauG, clean_adv, clean_nat, pgd_adv)

        mode = "EG"
        if best is None:
            # back to：E-only
            bestE = None
            for tauE in cand_E:
                clean_adv, clean_nat = self._rates(Es_c, Gs_c, tauE, float("+inf"))
                if clean_adv > self.clean_fpr_max: continue
                if self.router_logic.upper() == "AND" and (clean_nat > self.nat_fpr_max): continue
                pgd_adv, _ = self._rates(Es_a, Gs_a, tauE, float("+inf"))
                score = (abs(pgd_adv - self.target_adv_frac), -pgd_adv)
                if (bestE is None) or (score < bestE[0]):
                    bestE = (score, tauE, clean_adv, clean_nat, pgd_adv)
            # back to：G-only
            bestG = None
            for tauG in cand_G:
                clean_adv, clean_nat = self._rates(Es_c, Gs_c, float("-inf"), tauG)
                if clean_adv > self.clean_fpr_max: continue
                if self.router_logic.upper() == "AND" and (clean_nat > self.nat_fpr_max): continue
                pgd_adv, _ = self._rates(Es_a, Gs_a, float("-inf"), tauG)
                score = (abs(pgd_adv - self.target_adv_frac), -pgd_adv)
                if (bestG is None) or (score < bestG[0]):
                    bestG = (score, tauG, clean_adv, clean_nat, pgd_adv)

            if (bestE is not None) or (bestG is not None):
                use_E = (bestE is not None) and (bestG is None or bestE[0] < bestG[0])
                if use_E:
                    _, tauE, clean_adv, clean_nat, pgd_adv = bestE
                    tauG, mode = float("+inf"), "E-only"   # only E
                else:
                    _, tauG, clean_adv, clean_nat, pgd_adv = bestG
                    tauE, mode = float("-inf"), "G-only"   # only G
            else:
                # worst-case "safety"：clean 的 99% quantile
                tauE = float(torch.quantile(Es_c, torch.tensor(0.99)).item())
                tauG = float(torch.quantile(Gs_c, torch.tensor(0.99)).item())
                clean_adv, clean_nat = self._rates(Es_c, Gs_c, tauE, tauG)
                pgd_adv, _ = self._rates(Es_a, Gs_a, tauE, tauG)
                mode = "fallback-99"
        else:
            _, tauE, tauG, clean_adv, clean_nat, pgd_adv = best

        cfg.tE_abs = float(tauE)
        cfg.tG_abs = float(tauG)
        cfg.tE_quantile = None
        cfg.tG_quantile = None

        msg = (f"[Calib*] mode={mode} | logic={self.router_logic} "
            f"| tauE={cfg.tE_abs:.4f}, tauG={cfg.tG_abs:.4f} "
            f"| clean_adv={clean_adv:.2f}, clean_nat={clean_nat:.2f}, pgd_adv={pgd_adv:.2f} "
            f"(FPR_adv≤{self.clean_fpr_max}, FPR_nat≤{self.nat_fpr_max})")
        print(msg)

        if self.use_wandb and getattr(cfg, "use_wandb", False):
            import wandb
            wandb.log({
                f"{self.wandb_prefix}/mode": mode,
                f"{self.wandb_prefix}/router_logic": self.router_logic,
                f"{self.wandb_prefix}/tauE_abs": cfg.tE_abs,
                f"{self.wandb_prefix}/tauG_abs": cfg.tG_abs,
                f"{self.wandb_prefix}/clean_adv_rate": clean_adv,
                f"{self.wandb_prefix}/clean_nat_rate": clean_nat,
                f"{self.wandb_prefix}/pgd_adv_rate": pgd_adv,
                f"{self.wandb_prefix}/FPR_max": self.clean_fpr_max,
                f"{self.wandb_prefix}/FPR_nat_max": self.nat_fpr_max,
            })
        self.last = dict(mode=mode, tauE=cfg.tE_abs, tauG=cfg.tG_abs,
                         clean_adv=clean_adv, clean_nat=clean_nat, pgd_adv=pgd_adv)
        return cfg.tE_abs, cfg.tG_abs

    def maybe_update(self, model, cfg, epoch: int):
        if (epoch % self.interval) != 0:
            return None
        return self.calibrate(model, cfg)