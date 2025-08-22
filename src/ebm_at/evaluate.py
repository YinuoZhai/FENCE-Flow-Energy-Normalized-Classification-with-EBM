import torch
from torch.utils.data import DataLoader
from ebm_at.ebm import BackboneWithEBM
from ebm_at.utils import bn_eval
from ebm_at.routing import route_and_mask
from ebm_at.attacks import _pgd_adversary
from ebm_at.config import Config


def evaluate_clean(model: BackboneWithEBM, loader: DataLoader, cfg: Config) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            logits = model(x)["logits"]
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = 100.0 * correct / total
    print(f"[Eval] Clean Acc: {acc:.2f}%")
    return acc


def evaluate_routed(model: BackboneWithEBM, loader: DataLoader, cfg: Config) -> float:
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(cfg.device)
        y = y.to(cfg.device)
        with torch.enable_grad(), bn_eval(model):
            logits, diag = route_and_mask(model, x, cfg, y=y, backprop=False)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    acc = 100.0 * correct / total
    print(f"[Eval] Routed Acc (ID path + masks): {acc:.2f}%")
    return acc


def evaluate_robust_pgd(model, loader, cfg, steps=20):
    model.eval()
    correct = total = 0
    with bn_eval(model):
        for x, y in loader:
            x = x.to(cfg.device); y = y.to(cfg.device)
            x_adv = _pgd_adversary(model, x, y, cfg, steps=steps)
            with torch.no_grad():
                pred = model(x_adv)["logits"].argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = 100.0 * correct / total
    print(f"[Eval] Robust Acc (PGD-{steps}, no routing): {acc:.2f}%")
    return acc


def evaluate_routed_adv(model, loader, cfg, steps=20):
    model.eval()
    correct = total = 0
    n_id = n_nat = n_adv = 0
    M_means = []

    with bn_eval(model), torch.enable_grad():
        for x, y in loader:
            x = x.to(cfg.device); y = y.to(cfg.device)
            x_adv = _pgd_adversary(model, x, y, cfg, steps=steps)
            logits, diag = route_and_mask(model, x_adv, cfg, y=y, logic_override=cfg.router_logic_adv)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

            n_id  += int(diag["is_id"].sum().item())
            n_nat += int(diag["is_nat"].sum().item())
            n_adv += int(diag["is_adv"].sum().item())
            M_means.append(float(diag["M_mean"].item()))

    acc = 100.0 * correct / total
    frac_id  = n_id / total
    frac_nat = n_nat / total
    frac_adv = n_adv / total
    M_mean = sum(M_means) / max(1, len(M_means))

    print(f"[Eval] Routed Robust Acc (PGD-{steps}): {acc:.2f}% | mix: id={frac_id:.2f}, nat={frac_nat:.2f}, adv={frac_adv:.2f}, M_mean={M_mean:.3f}")
    return acc, {"frac_id": frac_id, "frac_nat": frac_nat, "frac_adv": frac_adv, "M_mean": M_mean}


def _extract_E_G(model, x):
    """get (E, G)：sample energy & gradient norm"""
    x = x.detach().requires_grad_(True)
    out = model(x)
    E = out["E_s"]                        # (B,)
    g = torch.autograd.grad(E.sum(), x, retain_graph=False, create_graph=False)[0]
    G = g.flatten(1).norm(dim=1)         # (B,)
    return E.detach().cpu(), G.detach().cpu()

@torch.no_grad()
def _quantile(t: torch.Tensor, q: float) -> float:
    q = float(max(0.0, min(1.0, q)))
    return torch.quantile(t, q).item()


def debug_check_grad_signal(model, loader, cfg, n_batches=2):
    model.eval()
    from math import isfinite
    with bn_eval(model), torch.enable_grad():
        for i, (x, y) in enumerate(loader):
            if i >= n_batches: break
            x = x.to(cfg.device).detach().requires_grad_(True)
            out = model(x)
            E = out["E_s"]
            print("[debug] E.requires_grad =", E.requires_grad, "| E.mean =", E.detach().mean().item())
            g = torch.autograd.grad(E.sum(), x, retain_graph=False, create_graph=False)[0]
            G = g.flatten(1).norm(1)
            print("[debug] G.mean =", float(G.mean()))