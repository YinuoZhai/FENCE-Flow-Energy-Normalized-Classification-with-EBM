import argparse
from ebm_at.config import Config
from ebm_at.utils import seed_everything
from ebm_at.data import get_cifar10_loaders
from ebm_at.utils import freeze_bn_running_stats
from ebm_at.ebm import BackboneWithEBM
from ebm_at.flow import VectorField
from ebm_at.train import train_ce_warmup, train_phase1_fm, train_phase2_joint
from ebm_at.calibrator import RouterCalibrator
from ebm_at.evaluate import evaluate_clean, evaluate_routed, evaluate_robust_pgd, evaluate_routed_adv
from ebm_at.evaluate import debug_check_grad_signal
from ebm_at.visual import viz_hist_and_scatter_EG, viz_energy_landscape_2d, viz_feature_scores_and_mask
 


def main():
    seed_everything(0)
    cfg = Config()

    print("Config:", cfg)
    if cfg.use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=cfg.__dict__)
        wandb.define_metric("epoch"); wandb.define_metric("iter_step")

    train_loader, test_loader = get_cifar10_loaders(cfg)
    model = BackboneWithEBM(num_classes=cfg.num_classes, feature_layer=cfg.feature_layer).to(cfg.device)

    viz_hist_and_scatter_EG(model, test_loader, cfg, max_batches=20)
    viz_energy_landscape_2d(model, next(iter(test_loader))[0], next(iter(test_loader))[1], cfg, grid=25, span=0.5)
    viz_feature_scores_and_mask(model, next(iter(test_loader))[0], cfg)

    # Phase-0 cross-entropy warm up
    train_ce_warmup(model, train_loader, cfg)
    acc0 = evaluate_clean(model, test_loader, cfg)

    # Phase-1 Flow matching
    vf = VectorField(z_dim=cfg.latent_dim, hidden=cfg.vf_hidden).to(cfg.device)
    train_phase1_fm(model, vf, train_loader, cfg)

    # RouterCalibrator
    calibrator = RouterCalibrator(
    calib_loader=test_loader,
    interval=5,           # every 5 epoch
    max_batches=20,
    pgd_steps=10,
    clean_fpr_max=0.05,
    nat_fpr_max = 0.05,
    target_adv_frac=0.70,
    q_grid=(0.60, 0.70, 0.80, 0.85, 0.90, 0.95),
    use_wandb=getattr(cfg, "use_wandb", False),
    )

    freeze_bn_running_stats(model)
    # Phase-2
    train_phase2_joint(model, train_loader, cfg, vf=vf,calibrator=calibrator)

    debug_check_grad_signal(model, test_loader, cfg, n_batches=2)

    viz_hist_and_scatter_EG(model, test_loader, cfg, max_batches=20)
    viz_energy_landscape_2d(model, next(iter(test_loader))[0], next(iter(test_loader))[1], cfg, grid=25, span=0.5)
    viz_feature_scores_and_mask(model, next(iter(test_loader))[0], cfg)

    acc_clean = evaluate_clean(model, test_loader, cfg)
    acc_routed = evaluate_routed(model, test_loader, cfg)
    rob_no_route = evaluate_robust_pgd(model, test_loader, cfg, steps=20)
    rob_routed, mix = evaluate_routed_adv(model, test_loader, cfg, steps=20)
    if cfg.use_wandb:
        wandb.log({
        "epoch/clean_acc": acc_clean,
        "epoch/routed_acc": acc_routed,
        "epoch/robust_acc_pgd20": rob_no_route,
        "epoch/routed_robust_acc_pgd20": rob_routed
        })
        wandb.finish()

if __name__ == "__main__":
    main()