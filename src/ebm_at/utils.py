import torch, random, numpy as np
from contextlib import contextmanager
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def to_device(x, device):
    return x.to(device, non_blocking=True)


def channels_last(model):
    return model.to(memory_format=torch.channels_last)


def set_amp():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def count_params(m): return sum(p.numel() for p in m.parameters())


def _set_bn_eval(m):
    if isinstance(m, nn.BatchNorm2d):
        m.eval()


@contextmanager
def bn_eval(model):
    # Temporarily set all BatchNorm layers to eval() (freeze running stats).
    was_training = model.training
    bns = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            bns.append((m, m.training))
    model.apply(_set_bn_eval)
    try:
        yield
    finally: # recover orginal BN training status
        for m, t in bns:
            m.train(t)
        model.train(was_training)


def freeze_bn_running_stats(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            

def set_requires_grad(params, flag: bool):
    for p in params:
        p.requires_grad_(flag)


def linear_anneal(epoch: int, start: int, end: int, max_val: float) -> float:
    if epoch < start: return 0.0
    if epoch >= end:  return max_val
    ratio = (epoch - start) / max(1, end - start)
    return max_val * float(ratio)


def ramp_int(epoch: int, start: int, end: int, v0: int, v1: int) -> int:
    if epoch <= start: return v0
    if epoch >= end:   return v1
    ratio = (epoch - start) / max(1, end - start)
    return int(round(v0 + ratio * (v1 - v0)))


class EnergyNormalizer(nn.Module):
    """Keep an EMA of (mu, std) and standardize E_s(x)."""
    def __init__(self, momentum=0.99, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("mu", torch.zeros(1))
        self.register_buffer("std", torch.ones(1))

    def forward(self, E: torch.Tensor, update: bool = True) -> torch.Tensor:
        # E: (B,)
        if self.training and update:
            with torch.no_grad():
                bmu = E.mean()
                bstd = E.std().clamp_min(self.eps)
                self.mu.mul_(self.momentum).add_(bmu * (1 - self.momentum))
                self.std.mul_(self.momentum).add_(bstd * (1 - self.momentum))
        return (E - self.mu) / (self.std + self.eps)


def boundary_energy_from_logits(
    logits: torch.Tensor, y: Optional[torch.Tensor],
    mode: str = "lsep", tau: float = 1.0
) -> torch.Tensor:
    """
    Classification boundary perception energy as:
      - "lsep":   E_cls = logsumexp(z/τ) - z_y/τ
      or
      - "neg_margin": E_cls = max_{k≠y} z_k - z_y

    logits: (B,C)
    y:      (B,) or None (will use argmax as pseudo-label)
    """
    z = logits.detach()
    if y is None:
        y = z.argmax(dim=1)
    if mode == "lsep":
        zt = z / tau
        lse = torch.logsumexp(zt, dim=1)
        zy  = zt.gather(1, y.view(-1, 1)).squeeze(1)
        return lse - zy
    elif mode == "neg_margin":
        zy = z.gather(1, y.view(-1, 1)).squeeze(1)
        zmax_oth = z.masked_fill(
            F.one_hot(y, num_classes=z.size(1)).bool(), float("-inf")
        ).max(dim=1).values
        return (zmax_oth - zy)
    else:
        raise ValueError(f"Unknown cls-energy mode: {mode}")
    



