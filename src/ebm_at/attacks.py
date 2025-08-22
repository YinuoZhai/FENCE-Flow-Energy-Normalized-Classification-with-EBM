import torch
import torch, torch.nn as nn
from ebm_at.utils import bn_eval
from ebm_at.config import Config

cfg = Config()

def pgd_attack(x: torch.Tensor, y: torch.Tensor, loss_fn, forward_logit_fn, eps: float, alpha: float, steps: int) -> torch.Tensor:
    """Simple untargeted PGD on the classification loss to craft adversarial examples.
    forward_logit_fn(x)->logits; loss_fn(logits, y)->scalar.
    """
    std = torch.tensor(cfg.data_std, device=x.device).view(1,3,1,1)
    eps_vec   = cfg.pgd_eps / std
    alpha_vec = cfg.pgd_alpha / std

    x_adv = x.detach() + 0.001 * torch.randn_like(x)
    x_adv = x_adv.clamp(-3, 3)  # assuming normalized input approx in [-3,3]
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = forward_logit_fn(x_adv)
        loss = loss_fn(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv + alpha_vec * torch.sign(grad)
        x_adv = torch.min(torch.max(x_adv, x - eps_vec), x + eps_vec)
        x_adv = x_adv.clamp(-3, 3).detach()
    return x_adv


def _pgd_adversary(model, x, y, cfg, steps=20, alpha=None):
    if alpha is None:
        alpha = cfg.pgd_eps / 4
    ce = nn.CrossEntropyLoss()
    def fwd(inp): return model(inp)["logits"]
    x_adv = x.detach() + 0.001 * torch.randn_like(x)
    x_adv = x_adv.clamp(-3, 3)  # assuming normalized input approx in [-3,3]
    for _ in range(steps):
        x_adv.requires_grad_(True)
        inp = x_adv.clone().detach().requires_grad_(True)
        with bn_eval(model), torch.enable_grad():
            logits = fwd(inp)
            loss = ce(logits, y)
        grad = torch.autograd.grad(loss, inp)[0]
        x_adv = inp + alpha * torch.sign(grad)
        x_adv = torch.min(torch.max(x_adv, x - cfg.pgd_eps), x + cfg.pgd_eps)
        x_adv = x_adv.clamp(-3, 3).detach()
    return x_adv


def _ensure_pgd(model, x, y, cfg, steps=10, alpha=None):
    """minimal PGD, feel free to use your own _pgd_adversary"""
    if alpha is None:
        alpha = cfg.pgd_eps / 4
    ce = nn.CrossEntropyLoss()
    x_adv = x.detach() + 0.001 * torch.randn_like(x)
    x_adv = x_adv.clamp(-3, 3)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        with bn_eval(model), torch.enable_grad():
            logits = model(x_adv)["logits"]
            loss = ce(logits, y)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        x_adv = x_adv + alpha * torch.sign(grad)
        x_adv = torch.min(torch.max(x_adv.detach(), x - cfg.pgd_eps), x + cfg.pgd_eps)
        x_adv = x_adv.clamp(-3, 3)
    return x_adv.detach()