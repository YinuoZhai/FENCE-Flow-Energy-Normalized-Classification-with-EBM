from dataclasses import dataclass
import torch
import yaml
import os
import importlib.util

@dataclass
class Config:
    data_root: str = "./data"
    batch_size: int = 512
    num_workers: int = 8
    num_classes: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimizer & training
    lr: float = 2e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 50   # cross-entropy (CE) warm up epoches
    epochs_phase1: int = 10   # latent FM warm-up epochs
    epochs_phase2: int = 50  # joint EBM training epochs

    # PGD (for adversarial negatives and robust training)
    pgd_eps: float = 8/255
    pgd_alpha: float = 2/255
    pgd_steps: int = 10

    # Phase-1: Flow Matching (latent)
    latent_dim: int = 512      # pooled feature dim (ResNet18 final stage = 512)
    vf_hidden: int = 512
    fm_noise_std: float = 0.0  # optional small noise on z_t for stability

    # DSM / EM (sample-level) losses
    dsm_sigma: float = 0.05
    em_langevin_steps: int = 3
    em_step_size: float = 0.05
    em_noise_std: float = 0.01

    # NCE (feature-level) losses
    nce_k: int = 5

    # weights
    w_ce: float = 1.0                  # cross-entropy
    w_dsm: float = 0.05                
    w_em: float = 0.05
    w_nce: float = 0.2
    w_order: float = 0.1               # Energy Ordering
    w_oe: float = 0.1                  # Outlier Exposure
    w_mask_l1: float = 1e-4
    w_advpush: float = 0.5

    # Feature-head & masking
    feature_layer: str = "layer3"  # which layer to use for feature-level EBM
    mask_gamma: float = 3.0
    mask_delta: float = 0.0
    mask_alpha_nat: float = 0.3     # keep small portion of non-robust feat for natural OOD
    score_alpha: float = 1.0        # weight for energy score term in q
    score_beta: float = 1.0         # weight for grad-norm term in q

    # Combined energy
    alpha_energy_fusion: float = 0.5   # E_total = α * E_norm + (1-α) * E_cls
    cls_energy_mode: str = "lsep"      # ["lsep", "neg_margin"]
    cls_tau: float = 0.7               # temperature for E_cls
    energy_norm_momentum: float = 0.99 # EMA for (μ,σ)
    energy_norm_eps: float = 1e-5

    # Routing thresholds (batch quantile fallback if None)
    tE_abs: float = 2.5
    tG_abs: float = 1.0
    tE_quantile: float = 0.9
    tG_quantile: float = 0.9

    router_logic_clean: str = "AND"
    router_logic_adv:   str = "OR"

    # cross-entropy warm up
    warmup_lr: float = 0.1
    warmup_use_sgd: bool = True     # True=SGD(m=0.9, wd=5e-4), False=Adam(lr=1e-3)


    em_px_max: float = 0.05        # (pixel) EM final max weight
    dsm_max: float   = 0.05        # DSM final max weight
    em_lat_max: float = 0.05       # latent-EM final max weight

    em_start: int = 3              #  Phase-2 temperature rise
    em_end: int   = 10
    dsm_start: int = 3
    dsm_end: int   = 10
    em_lat_start: int = 1
    em_lat_end: int   = 8

    pgd_steps_start: int = 1       # PGD linear temperature rise
    pgd_steps_end: int   = 5
    pgd_steps_end_epoch: int = 10  # before this epoch reaches pgd_steps_end

    # Flow negatives
    fm_lat_step: float = 0.1          # back step size
    fm_lat_K: int = 2             # k steps back
    use_flow_neg_in_nce: bool = True

    # temperature
    energy_tau: float = 10.0
    energy_center: bool = True

    # mean and std of CIFAR-10
    data_mean = [0.4914, 0.4822, 0.4465]
    data_std  = [0.2470, 0.2435, 0.2616]

    # Logging
    log_interval: int = 100

    use_wandb: bool = True
    wandb_project: str = "EBM_AT_project"
    wandb_run_name: str = "EBM_AT_running"

def load_config(path: str) -> Config:
    ext = os.path.splitext(path)[1]
    
    if ext in [".yaml", ".yml"]:
        with open(path, "r") as f:
            cfg_dict = yaml.safe_load(f)
        return cfg_dict   
    
    elif ext == ".py":
        spec = importlib.util.spec_from_file_location("config_module", path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        if hasattr(config_module, "Config"):
            return config_module.Config()  
        else:
            raise ValueError(f"{path} No Config class")
    else:
        raise ValueError(f"Unsupported format: {ext}")