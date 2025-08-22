from typing import List
import torch
import torch.nn.functional as F
from ebm_at.ebm import FeatureEnergyHead


def dsm_loss_sample_energy(E_s_fn, x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Denoising Score Matching loss on sample-level energy.
    L_DSM = E_{x,eps} || \nabla_x E(x+eps) + eps/sigma^2 ||^2
    """
    eps = sigma * torch.randn_like(x)
    x_noisy = x + eps
    x_noisy.requires_grad_(True)
    E = E_s_fn(x_noisy)  # (B,)
    grad = torch.autograd.grad(E.sum(), x_noisy, create_graph=False, retain_graph=False)[0]
    target = -eps / (sigma ** 2)
    return F.mse_loss(grad, target)


def langevin_negatives(x_init: torch.Tensor, energy_fn, steps: int, step_size: float, noise_std: float) -> torch.Tensor:
    """Generate negative samples by Langevin dynamics on energy E(x).
    x_{k+1} = x_k - step * grad E(x_k) + noise
    Detach trajectory (stop-grad) so we don't backprop through the sampler.
    """
    x = x_init.clone().detach()
    x.requires_grad_(True)
    for _ in range(steps):
        E = energy_fn(x)  # (B,)
        grad = torch.autograd.grad(E.sum(), x, create_graph=False, retain_graph=False)[0]
        x = x - step_size * grad + noise_std * torch.randn_like(x)
        x = x.detach()
        x.requires_grad_(True)
    return x.detach()


def energy_matching_loss(E_fn, x_pos: torch.Tensor, x_neg: torch.Tensor) -> torch.Tensor:
    """EM/CD-style loss: E(x_pos) - E(x_neg).
    """
    E_pos = E_fn(x_pos)
    E_neg = E_fn(x_neg)
    return (E_pos.mean() - E_neg.mean())


def nce_loss_feature_level(E_f_scalar: torch.Tensor, pos_feats: torch.Tensor, neg_feats_list: List[torch.Tensor]) -> torch.Tensor:
    """Feature-level NCE using energy as discriminator.
    For simplicity, we compute E_f on pos and multiple negs and apply logistic loss.

    Inputs:
      E_f_scalar: not used directly here (we recompute per negative); kept to show interface.
      pos_feats: (B,C,H,W)
      neg_feats_list: list of (B,C,H,W)
    """
    B = pos_feats.size(0)
    # A tiny discriminator using energy head semantics: we will use mean channel score as energy.
    # We'll implement a small inline energy function mirroring FeatureEnergyHead.conv+GAP+MLP.
    # However, to avoid weight mismatch, NCE should be computed via the actual feature head.
    # In practice, call the same head externally; here we provide a wrapper.
    raise NotImplementedError("Use nce_loss_feature_level_with_head to leverage the model's feature head.")


def nce_loss_feature_level_with_head(feature_head: FeatureEnergyHead, pos_feats: torch.Tensor, neg_feats_list: List[torch.Tensor]) -> torch.Tensor:
    """NCE using the provided feature_head to score energies.
    Logistic loss: maximize log sigma(-E(pos)) + sum log (1 - sigma(-E(neg)))
    Equivalently, minimizing: -[ log sigma(-E_pos) + sum log (1 - sigma(-E_neg)) ]
    """
    E_pos, _ = feature_head(pos_feats)  # (B,)
    logits_pos = -E_pos  # higher -> more data-like
    loss = F.binary_cross_entropy_with_logits(logits_pos, torch.ones_like(logits_pos))
    for neg in neg_feats_list:
        E_neg, _ = feature_head(neg)
        logits_neg = -E_neg
        loss += F.binary_cross_entropy_with_logits(logits_neg, torch.zeros_like(logits_neg))
    return loss / (1 + len(neg_feats_list))


def energy_ordering_loss(E_clean: torch.Tensor, E_aug: torch.Tensor, E_adv: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """Enforce E(clean) < E(aug) < E(adv) with margins: hinge-style.
    """
    loss1 = F.relu(margin + E_clean - E_aug).mean()
    loss2 = F.relu(margin + E_aug - E_adv).mean()
    return loss1 + loss2


def oe_margin_loss(E_clean: torch.Tensor, E_oe: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """Outlier Exposure margin: E(oe) - E(clean) >= margin.
    """
    return F.relu(margin - (E_oe - E_clean)).mean()


def adv_push_loss(E_clean, E_adv, tau=1.0):
    return torch.nn.functional.softplus((E_clean - E_adv) / tau).mean()