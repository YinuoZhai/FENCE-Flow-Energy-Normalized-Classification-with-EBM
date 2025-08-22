import torch
import torch.nn as nn
from ebm_at.utils import bn_eval
from ebm_at.routing import  feature_channel_mask
from ebm_at.attacks import _pgd_adversary
import matplotlib.pyplot as plt


@torch.no_grad()
def viz_hist_and_scatter_EG(model, loader, cfg, max_batches=20, out_prefix="viz_eg"):
    model.eval()
    Es_c, Gs_c, Es_a, Gs_a = [], [], [], []
    with bn_eval(model), torch.enable_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches: break
            x = x.to(cfg.device); y = y.to(cfg.device)
            # clean
            xc = x.detach().requires_grad_(True)
            Ec = model.combined_energy_from_out(model(xc), y=y, alpha=cfg.alpha_energy_fusion,
                                     update_norm=False, cls_mode=cfg.cls_energy_mode, cls_tau=cfg.cls_tau)
            gc = torch.autograd.grad(Ec.sum(), xc, retain_graph=False, create_graph=False)[0]
            Gc = gc.flatten(1).norm(dim=1)
            Es_c.append(Ec.detach().cpu()); Gs_c.append(Gc.detach().cpu())
            # pgd
            xa = _pgd_adversary(model, x, y, cfg, steps=10)
            xa = xa.detach().requires_grad_(True)
            Ea = model.combined_energy_from_out(model(xa), y=y, alpha=cfg.alpha_energy_fusion,
                                     update_norm=False, cls_mode=cfg.cls_energy_mode, cls_tau=cfg.cls_tau)
            ga = torch.autograd.grad(Ea.sum(), xa, retain_graph=False, create_graph=False)[0]
            Ga = ga.flatten(1).norm(dim=1)
            Es_a.append(Ea.detach().cpu()); Gs_a.append(Ga.detach().cpu())

    Es_c = torch.cat(Es_c); Gs_c = torch.cat(Gs_c)
    Es_a = torch.cat(Es_a); Gs_a = torch.cat(Gs_a)


    fig1, axs = plt.subplots(1, 2, figsize=(10,4))
    axs[0].hist(Es_c.numpy(), bins=50, alpha=0.5, label='clean')
    axs[0].hist(Es_a.numpy(), bins=50, alpha=0.5, label='pgd')
    axs[0].set_title("Energy E(x)")
    axs[0].legend()

    axs[1].hist(Gs_c.numpy(), bins=50, alpha=0.5, label='clean')
    axs[1].hist(Gs_a.numpy(), bins=50, alpha=0.5, label='pgd')
    axs[1].set_title("Grad-norm G(x)=||∂E/∂x||")
    axs[1].legend()
    fig1.tight_layout()


    fig2, ax = plt.subplots(figsize=(5,4))
    ax.scatter(Es_c.numpy(), Gs_c.numpy(), s=3, alpha=0.4, label='clean')
    ax.scatter(Es_a.numpy(), Gs_a.numpy(), s=3, alpha=0.4, label='pgd')
    ax.set_xlabel("E"); ax.set_ylabel("G")
    ax.set_title("E–G scatter")
    ax.legend()
    fig2.tight_layout()

    if getattr(cfg, "use_wandb", False):
        import wandb
        wandb.log({"viz/EG_hist": wandb.Image(fig1), "viz/EG_scatter": wandb.Image(fig2)})
        plt.close(fig1); plt.close(fig2)
    else:
        fig1.savefig(f"{out_prefix}_hist.png", dpi=150); fig2.savefig(f"{out_prefix}_scatter.png", dpi=150)
        print(f"[viz] saved {out_prefix}_hist.png / {out_prefix}_scatter.png")


def _orthonormalize(v1):
    v1 = v1 / (v1.norm() + 1e-8)
    v2 = torch.randn_like(v1)
    v2 = v2 - (v2 * v1).sum() * v1
    v2 = v2 / (v2.norm() + 1e-8)
    return v1, v2


@torch.no_grad()
def viz_energy_landscape_2d(model, x, y, cfg, grid=25, span=0.5, steps_for_dir=10, out_prefix="viz_land2d"):

    model.eval()
    B = x.size(0)
    assert B == y.size(0)
    x = x.to(cfg.device); y = y.to(cfg.device)

    x0 = x[:1].clone().detach().requires_grad_(True)
    ce = nn.CrossEntropyLoss()
    with bn_eval(model), torch.enable_grad():
        logits = model(x0)["logits"]; loss = ce(logits, y[:1])
    g = torch.autograd.grad(loss, x0)[0]
    u = g.sign().flatten()                 # FGSM direction
    v = torch.randn_like(u)
    v = v - (v * u).sum() * u
    v = v / (v.norm() + 1e-8)

    a_vals = torch.linspace(-span, span, grid, device=cfg.device)
    b_vals = torch.linspace(-span, span, grid, device=cfg.device)
    Z = torch.zeros(grid, grid, device=cfg.device)

    base = x0.detach().flatten()
    for i, a in enumerate(a_vals):
        for j, b in enumerate(b_vals):
            x_ab = base + (a * cfg.pgd_eps) * u + (b * cfg.pgd_eps) * v
            x_ab = x_ab.view_as(x0).clamp(-3, 3)
            x_ab.requires_grad_(True)
            with bn_eval(model), torch.enable_grad():
                E = model(x_ab)["E_s"]  # (1,)
            Z[i, j] = E.detach()

    Z_cpu = Z.cpu().numpy()
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.contourf(a_vals.cpu().numpy(), b_vals.cpu().numpy(), Z_cpu.T, levels=30)
    ax.set_xlabel("a (FGSM dir)"); ax.set_ylabel("b (random orth dir)")
    ax.set_title("Energy landscape around x")
    fig.colorbar(im, ax=ax)

    if getattr(cfg, "use_wandb", False):
        import wandb
        wandb.log({"viz/landscape2d": wandb.Image(fig)})
        plt.close(fig)
    else:
        fig.savefig(f"{out_prefix}.png", dpi=150)
        print(f"[viz] saved {out_prefix}.png")


@torch.no_grad()
def viz_feature_scores_and_mask(model, x, cfg, out_prefix="viz_feat"):

    model.eval()
    x = x.to(cfg.device)
    with bn_eval(model), torch.enable_grad():
        out = model(x)
        f = out["feats"][cfg.feature_layer]
        M, q = feature_channel_mask(model.feature_head, f, cfg)  # (B,C), (B,C)

    q_all = q.flatten().detach().cpu().numpy()
    M_all = M.flatten().detach().cpu().numpy()

    fig1, axs = plt.subplots(1,2, figsize=(10,4))
    axs[0].hist(q_all, bins=50, alpha=0.8); axs[0].set_title("q scores")
    axs[1].hist(M_all, bins=50, alpha=0.8); axs[1].set_title("mask M")
    fig1.tight_layout()

    q0 = q[0].detach().cpu().numpy()
    idx = q0.argsort()[::-1][:16]
    fig2, ax = plt.subplots(figsize=(6,4))
    ax.bar(range(16), q0[idx])
    ax.set_xticks(range(16)); ax.set_xticklabels(idx, rotation=90)
    ax.set_title("Top-16 channels by q (sample 0)")
    fig2.tight_layout()

    if getattr(cfg, "use_wandb", False):
        import wandb
        wandb.log({"viz/q_M_hist": wandb.Image(fig1), "viz/q_topk": wandb.Image(fig2)})
        plt.close(fig1); plt.close(fig2)
    else:
        fig1.savefig(f"{out_prefix}_hist.png", dpi=150)
        fig2.savefig(f"{out_prefix}_topk.png", dpi=150)
        print(f"[viz] saved {out_prefix}_hist.png / {out_prefix}_topk.png")