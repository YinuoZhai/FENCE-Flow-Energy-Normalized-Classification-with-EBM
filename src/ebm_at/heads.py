from typing import Tuple
import torch, torch.nn as nn, torch.nn.functional as F

class SampleEnergyHead(nn.Module):
    """Sample-level energy head g_theta: takes pooled global feature -> scalar energy.
    Maps BxD -> Bx1. This is E_s(x).
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.SiLU(),
            nn.Linear(in_dim, 1)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, D)
        E = self.net(z).squeeze(-1)  # (B,)
        return E


class FeatureEnergyHead(nn.Module):
    """Feature-level energy head h_phi^(ell):
    - 1x1 conv to keep channel dimension (or project),
    - Global Average Pool (per-channel) -> u in R^C,
    - Per-channel linear gating (like SE) to produce channel scores s in R^C,
    - Aggregate energy E_f = mean(s) (scalar), used for grad wrt feature map.

    Also returns the per-channel s for mask computation.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.SiLU(),
            nn.Linear(in_channels // 2, in_channels)
        )

    def forward(self, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # f: (B, C, H, W)
        h = self.conv(f)  # (B,C,H,W)
        u = F.adaptive_avg_pool2d(h, 1).squeeze(-1).squeeze(-1)  # (B,C)
        s = self.mlp(u)  # (B,C) channel-wise scores
        E_f = s.mean(dim=1)  # (B,) scalar energy per sample
        return E_f, s


class ClassifierHead(nn.Module):
    """Classifier head operating on the final masked pooled feature.
    """
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.fc(pooled)
    

