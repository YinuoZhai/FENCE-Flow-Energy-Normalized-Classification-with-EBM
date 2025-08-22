from typing import Optional, Tuple, Dict
import torch, torch.nn.functional as F
from ebm_at.ebm import BackboneWithEBM
from ebm_at.ebm import FeatureEnergyHead
from ebm_at.config import Config


cfg = Config()


def robust_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # normalize per-sample vector
    mu = x.mean(dim=-1, keepdim=True)
    sd = x.std(dim=-1, keepdim=True) + eps
    return (x - mu) / sd


def compute_sample_energy_and_grad(model: BackboneWithEBM, x: torch.Tensor,
                                   y: Optional[torch.Tensor] = None,
                                   update_norm: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    x = x.detach().requires_grad_(True)
    out = model(x)
    # --- use combined energy ---
    E = model.combined_energy_from_out(
        out, y=y, alpha=cfg.alpha_energy_fusion,
        update_norm=update_norm, cls_mode=cfg.cls_energy_mode, cls_tau=cfg.cls_tau
    )  # (B,)
    g = torch.autograd.grad(E.sum(), x, retain_graph=True, create_graph=False)[0]
    G = g.flatten(1).norm(dim=1)  # (B,)
    return E.detach(), G.detach(), out


def feature_channel_mask(feature_head: FeatureEnergyHead, f: torch.Tensor, cfg: Config) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute channel-wise soft mask M using energy score and per-channel grad norm.
    Returns (M in [0,1], q raw score).
    """
    f = f.detach().requires_grad_(True)
    E_f, s_ch = feature_head(f)  # E_f: (B,), s_ch: (B,C)
    gf = torch.autograd.grad(E_f.sum(), f, retain_graph=False, create_graph=False)[0]  # (B,C,H,W)
    gf_ch = gf.pow(2).sum(dim=(2,3)).sqrt()  # (B,C)

    s_n = robust_normalize(s_ch)
    g_n = robust_normalize(gf_ch)
    q = cfg.score_alpha * s_n + cfg.score_beta * g_n

    M = torch.sigmoid(cfg.mask_gamma * q + cfg.mask_delta)
    return M.detach(), q.detach()


def route_and_mask(model: BackboneWithEBM, x: torch.Tensor, cfg: Config,
                   y=None, tE: Optional[float] = None, tG: Optional[float] = None, backprop: bool=False,
                   logic_override: Optional[str]=None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    """Compute routing decision using (E, G). Apply feature masks accordingly and produce logits.
    Returns logits and a dict of diagnostics.
    """
    logic = (logic_override or getattr(cfg, "router_logic", "AND")).upper()
    E, G, out = compute_sample_energy_and_grad(model, x, y=y, update_norm=False)
    feats = out["feats"][cfg.feature_layer]
    M, q = feature_channel_mask(model.feature_head, feats, cfg)

    tE = getattr(cfg, "tE_abs", None)
    tG = getattr(cfg, "tG_abs", None)

    # thresholds (quantiles over batch if not provided)
    if tE is None:
        tE = getattr(cfg, "tE_abs", None)
    if tG is None:
        tG = getattr(cfg, "tG_abs", None)

    if tE is None:
        qe = getattr(cfg, "tE_quantile", 0.9)
        tE = float(torch.quantile(E.detach(), qe).item())
    if tG is None:
        qg = getattr(cfg, "tG_quantile", 0.9)
        tG = float(torch.quantile(G.detach(), qg).item())

    # routing masks
    logic = getattr(cfg, "router_logic", "AND").upper()
    if logic == "OR":
        is_adv = (E > tE) | (G > tG)
        is_nat = (E > tE) & ~(G > tG)
        is_id  = ~(is_adv)
    else:
        is_adv = (E > tE) & (G > tG)
        is_nat = (E > tE) & (~is_adv)
        is_id  = ~(is_nat | is_adv)

    B, C, H, W = feats.shape
    feats_masked = feats.clone()
    if is_nat.any():
        if getattr(cfg, "mask_alpha_nat", 0.0) > 1e-8:
            alpha = cfg.mask_alpha_nat
            M_nat = M[is_nat].unsqueeze(-1).unsqueeze(-1)         # (Bn,C,1,1)
            f_nat = feats[is_nat]
            feats_nat = (1.0 - alpha) * f_nat + alpha * (M_nat * f_nat)
            feats_masked = feats_masked.clone()
            feats_masked[is_nat] = feats_nat
        else:
            pass

    if is_adv.any():
        M_adv = M[is_adv].unsqueeze(-1).unsqueeze(-1)            # (Ba,C,1,1)
        feats_masked = feats_masked.clone()
        feats_masked[is_adv] = M_adv * feats[is_adv]              # Hard mask: Only robust channels retained.

    # Recompute pooled features and logits from masked features: replace layer output, continue forward tail
    # We will manually forward tail (layer4 -> avgpool -> classifier)
    if backprop:
        l4 = model.layer4(feats_masked)
        pooled = model.avgpool(l4).view(B, -1)
    else:
      with torch.no_grad():
          # propagate masked features through remaining layers
          l4 = model.layer4(feats_masked)
          pooled = model.avgpool(l4).view(B, -1)
    logits = model.classifier(pooled)

    diag = {
        "E": E.detach(), "G": G.detach(),
        "tE": torch.tensor([tE], device=E.device),
        "tG": torch.tensor([tG], device=E.device),
        "is_id": is_id.float(), "is_nat": is_nat.float(), "is_adv": is_adv.float(),
        "M_mean": M.mean().detach(), "q_mean": q.mean().detach(),
        "router_logic": torch.tensor([1.0 if logic == "OR" else 0.0], device=E.device),
    }
    return logits, diag