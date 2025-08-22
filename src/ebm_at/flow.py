import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from ebm_at.config import Config
from ebm_at.ebm import BackboneWithEBM


class VectorField(nn.Module):
    """Simple MLP vector field v_phi([z, t]) -> velocity in latent space.
    Used for Rectified/Linear Flow Matching between prior z0~N(0,I) and data z1.
    """
    def __init__(self, z_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, z_dim)
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # z: (B,D), t: (B,1)
        inp = torch.cat([z, t], dim=1)
        v = self.net(inp)
        return v


def fm_training_step(model: BackboneWithEBM, vf: VectorField, x: torch.Tensor, opt_vf: torch.optim.Optimizer, cfg: Config) -> Dict[str, float]:
    """One FM step: project to latent (pooled features), sample pairs (z0,z1),
    linearly interpolate z_t and train v_phi(z_t,t) to match target (z1 - z0).
    """
    model.eval()  # freeze backbone for FM warm-up
    with torch.no_grad():
        out = model(x)
        z1 = out["pooled"]  # (B,D)

    B, D = z1.shape
    z0 = torch.randn_like(z1)
    t = torch.rand(B, 1, device=z1.device)
    z_t = (1 - t) * z0 + t * z1
    if cfg.fm_noise_std > 0:
        z_t = z_t + cfg.fm_noise_std * torch.randn_like(z_t)
    target_v = (z1 - z0)  # rectified flow target

    vf.train()
    pred_v = vf(z_t, t)
    loss = F.mse_loss(pred_v, target_v)

    opt_vf.zero_grad(set_to_none=True)
    loss.backward()
    opt_vf.step()
    return {"fm_loss": loss.item()}


@torch.no_grad()
def flow_step_from_data(z1: torch.Tensor, vf: nn.Module, step: float, K: int) -> torch.Tensor:
    """
    K steps back to the prior form z1： z <- z - step * v_phi(z, t=1)

    """
    z = z1.clone()
    B = z.size(0)
    for _ in range(K):
        t = torch.ones(B, 1, device=z.device)
        v = vf(z, t)      # approximately (z1 - z0) direction
        z = z - step * v  # one step backward
    return z