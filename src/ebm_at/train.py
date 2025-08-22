import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ebm_at.utils import bn_eval, set_requires_grad, linear_anneal, ramp_int
from ebm_at.ebm import BackboneWithEBM
from ebm_at.flow import VectorField, fm_training_step, flow_step_from_data
from ebm_at.loss import nce_loss_feature_level_with_head, energy_ordering_loss, oe_margin_loss
from ebm_at.loss import  dsm_loss_sample_energy, langevin_negatives, adv_push_loss
from ebm_at.routing import route_and_mask, feature_channel_mask
from ebm_at.calibrator import RouterCalibrator
from ebm_at.attacks import pgd_attack
from ebm_at.config import Config
import wandb

cfg = Config()

def train_ce_warmup(model: BackboneWithEBM, train_loader: DataLoader, cfg: Config):

    model.train()

    params = list(model.parameters())
    if cfg.warmup_use_sgd:
        opt = torch.optim.SGD(params, lr=cfg.warmup_lr, momentum=0.9, weight_decay=5e-4)
    else:
        opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)

    ce_loss = nn.CrossEntropyLoss()

    for ep in range(cfg.warmup_epochs):
        running = 0.0
        for it, (x, y) in enumerate(train_loader):
            x = x.to(cfg.device); y = y.to(cfg.device)
            out = model(x)
            logits = out["logits"]
            loss = ce_loss(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += loss.item()
            if (it + 1) % cfg.log_interval == 0:
                avg = running / cfg.log_interval
                print(f"[CE-warmup] Ep {ep+1}/{cfg.warmup_epochs} it {it+1}: ce={avg:.3f}")
                if hasattr(cfg, "use_wandb") and cfg.use_wandb:
                    wandb.log({"iter/ce_warmup": avg})
                running = 0.0
        print(f"[CE-warmup] Epoch {ep+1} done.")


def train_phase1_fm(model: BackboneWithEBM, vf: VectorField, train_loader: DataLoader, cfg: Config):
    opt_vf = torch.optim.Adam(vf.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    model.eval()
    vf.train()
    global_iter = 0
    for epoch in range(cfg.epochs_phase1):
        avg = 0.0
        for it, (x, y) in enumerate(train_loader):
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            stats = fm_training_step(model, vf, x, opt_vf, cfg)
            avg += stats["fm_loss"]
            global_iter += 1
            if cfg.use_wandb and (it + 1) % cfg.log_interval == 0:
                print(f"[FM] Epoch {epoch+1}/{cfg.epochs_phase1} Iter {it+1}: fm_loss={avg/cfg.log_interval:.4f}")
                wandb.log({
                    "iter/fm_loss": avg / cfg.log_interval,
                    "iter_step": global_iter,
                    "epoch": epoch + 1
                })
                avg = 0.0
        if cfg.use_wandb:
            wandb.log({"epoch/fm_last": stats["fm_loss"], "epoch": epoch + 1})

    print("[FM] Phase-1 complete.")


def train_phase2_joint(model: BackboneWithEBM, train_loader: DataLoader, cfg: Config, vf: nn.Module = None, calibrator: RouterCalibrator=None):
    """
    Phase-2 (anneal + flow):
      - Step-A(EBM): freeze backbone, only updates energy heads
      - Step-B(CE): only updates backbone+classifier
    """
    # === make sure flow_proj has（D -> C）===
    if not hasattr(model, "flow_proj"):
        # C: number of channels of feature_head ；D: pooled/latent dim
        C = model.feature_head.mlp[0].in_features
        D = model.sample_head.net[0].in_features
        model.flow_proj = nn.Linear(D, C).to(cfg.device)

    # params
    energy_params = list(model.sample_head.parameters()) + list(model.feature_head.parameters()) + list(model.flow_proj.parameters())
    main_params   = []
    for name, p in model.named_parameters():
        if ("sample_head" in name) or ("feature_head" in name) or ("flow_proj" in name):
            continue
        main_params.append(p)

    opt_ebm = torch.optim.Adam(energy_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    opt_ce  = torch.optim.SGD(main_params, lr=0.05, momentum=0.9, weight_decay=5e-4)
    ce_loss_fn = nn.CrossEntropyLoss()


    def _temp(E):
        E = E - E.mean() if getattr(cfg, "energy_center", True) else E
        return E / getattr(cfg, "energy_tau", 10.0)

    for epoch in range(cfg.epochs_phase2):

        w_em_px   = linear_anneal(epoch, cfg.em_start,   cfg.em_end,   cfg.em_px_max)
        w_dsm     = linear_anneal(epoch, cfg.dsm_start,  cfg.dsm_end,  cfg.dsm_max)
        w_em_lat  = linear_anneal(epoch, cfg.em_lat_start, cfg.em_lat_end, cfg.em_lat_max)
        pgd_steps = ramp_int(epoch, cfg.pgd_steps_start, cfg.pgd_steps_end_epoch,
                             cfg.pgd_steps_start, cfg.pgd_steps_end)

        model.train()
        running = {k: 0.0 for k in ["loss","ce","dsm","em_px","em_lat","nce","order","oe","mask_l1",
                                    "E_clean","E_adv","E_aug"]}

        for it, (x, y) in enumerate(train_loader):
            x = x.to(cfg.device); y = y.to(cfg.device)

            # generate light PGD samples
            model.zero_grad(set_to_none=True)
            with bn_eval(model), torch.enable_grad():
                x_adv = pgd_attack(
                    x, y, loss_fn=ce_loss_fn,
                    forward_logit_fn=lambda inp: model(inp)["logits"],
                    eps=cfg.pgd_eps, alpha=cfg.pgd_alpha, steps=pgd_steps,
                )
            model.zero_grad(set_to_none=True)

            # Step-A: update EBM only
            # Allow gradients to flow through backbone for EBM loss calculation
            set_requires_grad(main_params, False)  # Freeze backbone
            set_requires_grad(energy_params, True) # Ensure energy params require grad

            with bn_eval(model):
                # Forward passes with gradient tracking
                out_c = model(x)
                out_a = model(x_adv)

                feats_pos = out_c["feats"][cfg.feature_layer].detach() # Do not detach here
                feats_adv = out_a["feats"][cfg.feature_layer].detach() # Do not detach here

                pooled_c  = out_c["pooled"].detach()
                pooled_a  = out_a["pooled"].detach()

                # natural shift
                x_aug = (x + 0.05 * torch.randn_like(x)).clamp(-3, 3)
                x_oe  = (x + 0.15 * torch.randn_like(x)).clamp(-3, 3)

                out_aug = model(x_aug); out_oe = model(x_oe)
                pooled_aug = out_aug["pooled"].detach()
                pooled_oe  = out_oe["pooled"].detach()

                # sample energy
                E_clean = model.combined_energy_from_out(out_c,  y=y, alpha=cfg.alpha_energy_fusion,
                                             update_norm=True, cls_mode=cfg.cls_energy_mode, cls_tau=cfg.cls_tau)
                E_adv   = model.combined_energy_from_out(out_a,  y=y, alpha=cfg.alpha_energy_fusion,
                                             update_norm=True, cls_mode=cfg.cls_energy_mode, cls_tau=cfg.cls_tau)
                E_aug   = model.combined_energy_from_out(out_aug,y=y, alpha=cfg.alpha_energy_fusion,
                                             update_norm=True, cls_mode=cfg.cls_energy_mode, cls_tau=cfg.cls_tau)
                E_oe    = model.combined_energy_from_out(out_oe, y=y, alpha=cfg.alpha_energy_fusion,
                                             update_norm=True, cls_mode=cfg.cls_energy_mode, cls_tau=cfg.cls_tau)

                # ---- feature-level NCE ----
                # Detach negatives for NCE to avoid backprop through their generation
                neg_feats_list = [feats_adv.detach()]
                idx = torch.randperm(x.size(0), device=x.device)
                feats_shuf = feats_pos[idx].detach()
                neg_feats_list.append(feats_shuf)

                # flow negative examples：make z1 "one step back"，and remap back to the “channel score” space as a difficult example.
                if vf is not None and cfg.use_flow_neg_in_nce:
                    with torch.no_grad(): # Detach z1 and z_minus
                        z1 = pooled_c.detach()
                        z_minus = flow_step_from_data(z1, vf, cfg.fm_lat_step, cfg.fm_lat_K)  # (B,D)
                    # Project z_minus to the feature head MLP input dimension and pass to MLP
                    # s_flow requires grad with respect to flow_proj and feature_head.mlp params
                    s_flow = model.flow_proj(z_minus)                   # (B, C)
                    H, W = feats_pos.shape[2], feats_pos.shape[3]
                    s_flow_map = s_flow.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)  # (B,C,H,W)
                    neg_feats_list.append(s_flow_map.detach())

                nce = nce_loss_feature_level_with_head(model.feature_head, feats_pos, neg_feats_list)

                # ---- Energy Ordering & Outlier Exposure ----
                order = energy_ordering_loss(_temp(E_clean), _temp(E_aug), _temp(E_adv), margin=1.0)
                oe    = oe_margin_loss(_temp(E_clean), _temp(E_oe), margin=1.0)

                # ---- DSM（sample-level）----
                dsm = torch.tensor(0.0, device=x.device)
                if w_dsm > 0.0:
                    dsm = dsm_loss_sample_energy(
                        lambda inp: model.combined_energy_from_out(
                            model(inp), y=None, alpha=cfg.alpha_energy_fusion,
                            update_norm=False, cls_mode=cfg.cls_energy_mode, cls_tau=cfg.cls_tau
                        ),
                        x, sigma=cfg.dsm_sigma
                    )

                # ---- EM（pixel；Langevin steps）----
                em_px = torch.tensor(0.0, device=x.device)
                if w_em_px > 0.0:
                    x_noise = torch.randn_like(x)
                    neg = langevin_negatives(
                        x_init=x_noise,
                        energy_fn=lambda inp: model.combined_energy_from_out(
                            model(inp), y=None, alpha=cfg.alpha_energy_fusion,
                            update_norm=False, cls_mode=cfg.cls_energy_mode, cls_tau=cfg.cls_tau
                        ),
                        steps=cfg.em_langevin_steps, step_size=cfg.em_step_size, noise_std=cfg.em_noise_std
                    )
                    E_neg = model.combined_energy_from_out(
                        model(neg), y=None, alpha=cfg.alpha_energy_fusion,
                        update_norm=False, cls_mode=cfg.cls_energy_mode, cls_tau=cfg.cls_tau
                    )
                    em_px = (E_clean.mean() - E_neg.mean())

                # ---- EM（latent；make z1 "K step back" using vector field (vf)）----
                em_lat = torch.tensor(0.0, device=x.device)
                if (vf is not None) and (w_em_lat > 0.0):
                    with torch.no_grad(): # Detach z1 and z_minus
                        z1 = pooled_c.detach()
                        z_minus = flow_step_from_data(z1, vf, cfg.fm_lat_step, cfg.fm_lat_K)  # (B,D)
                    # E requires grad w.r.t sample_head params
                    E_pos_z = model.sample_head(z1)
                    E_neg_z = model.sample_head(z_minus)
                    E_pos_z = model.energy_norm(E_pos_z, update=True)
                    E_neg_z = model.energy_norm(E_neg_z, update=False)
                    em_lat = (E_pos_z.mean() - E_neg_z.mean())

                L_advpush = adv_push_loss(E_clean.detach(), E_adv, tau=1.0)

                ebm_total = (cfg.w_nce * nce +
                             cfg.w_order * order +
                             cfg.w_oe * oe +
                             w_dsm * dsm +
                             w_em_px * em_px +
                             w_em_lat * em_lat +
                             cfg.w_advpush * L_advpush)


                energy_l2 = 1e-6 * (E_clean.pow(2).mean() + E_adv.pow(2).mean()) # L2 regularization
                ebm_total = ebm_total + energy_l2

                opt_ebm.zero_grad(set_to_none=True)
                ebm_total.backward()
                torch.nn.utils.clip_grad_norm_(energy_params, 5.0)
                opt_ebm.step()

            # Zero out gradients for main params after EBM step
            for p in main_params:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()


            # Step-B: update backbone+classifier（cross-entropy step）only

            set_requires_grad(main_params, True)
            set_requires_grad(energy_params, False) # Freeze energy heads for CE step

            # clean pass（allow BatchNorm updates using clean ）
            out_clean2 = model(x)
            logits_clean = out_clean2["logits"]

            # adv router（BN frozen；Allow gradient calculation for x but do not update the energy head.）
            with bn_eval(model), torch.enable_grad():
                 # Ensure feats_pos is not detached for feature_channel_mask
                feats_pos_for_mask = model(x)["feats"][cfg.feature_layer]
                logits_adv_masked, _ = route_and_mask(model, x_adv, cfg, backprop=True, logic_override=cfg.router_logic_adv)


            ce = 0.5 * (ce_loss_fn(logits_clean, y) + ce_loss_fn(logits_adv_masked, y))

            # mask L1（using feats_pos from clean pass）
            M, _ = feature_channel_mask(model.feature_head, feats_pos_for_mask.detach(), cfg)
            mask_l1 = M.mean()


            total = (cfg.w_ce * ce +
                     cfg.w_mask_l1 * mask_l1 +
                     # These terms were used in Step-A to update energy heads,
                     # include their detached values for logging total loss
                     (cfg.w_nce * nce.detach()) +
                     (cfg.w_order * order.detach()) +
                     (cfg.w_oe * oe.detach()) +
                     (w_dsm * dsm.detach()) +
                     (w_em_px * em_px.detach()) +
                     (w_em_lat * em_lat.detach()) +
                     (cfg.w_advpush * L_advpush.detach()))

            opt_ce.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(main_params, 5.0)
            opt_ce.step()


            running["loss"] += float(total.item())
            running["ce"]   += float(ce.item())
            running["nce"]  += float(nce.item())
            running["order"]+= float(order.item())
            running["oe"]   += float(oe.item())
            running["dsm"]  += float(dsm.item())
            running["em_px"]+= float(em_px.item())
            running["em_lat"]+= float(em_lat.item())
            running["mask_l1"] += float(mask_l1.item())
            running["E_clean"] += float(E_clean.detach().mean().item())
            running["E_adv"]   += float(E_adv.detach().mean().item())
            running["E_aug"]   += float(E_aug.detach().mean().item())

            if getattr(cfg, "use_wandb", False) and ((it + 1) % cfg.log_interval == 0):
                import wandb
                n = cfg.log_interval
                wandb.log({
                    "iter/total": running["loss"]/n,
                    "iter/ce": running["ce"]/n,
                    "iter/nce": running["nce"]/n,
                    "iter/order": running["order"]/n,
                    "iter/oe": running["oe"]/n,
                    "iter/dsm": running["dsm"]/n,
                    "iter/em_px": running["em_px"]/n,
                    "iter/em_lat": running["em_lat"]/n,
                    "iter/mask_l1": running["mask_l1"]/n,
                    "iter/E_clean_mean": running["E_clean"]/n,
                    "iter/E_adv_mean": running["E_adv"]/n,
                    "iter/E_aug_mean": running["E_aug"]/n,
                    "iter/pgd_steps": pgd_steps,
                    "iter/w_em_px": w_em_px,
                    "iter/w_dsm": w_dsm,
                    "iter/w_em_lat": w_em_lat,
                    "epoch": epoch + 1
                })
                for k in running: running[k] = 0.0

        if calibrator is not None:
            calibrator.maybe_update(model, cfg, epoch=epoch+1)

        print(f"[P2] Epoch {epoch+1} done.")