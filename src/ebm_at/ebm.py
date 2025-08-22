from typing import Dict, Optional
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models
from ebm_at.heads import SampleEnergyHead, FeatureEnergyHead, ClassifierHead
from ebm_at.config import Config
from ebm_at.utils import EnergyNormalizer, boundary_energy_from_logits


cfg = Config()

class BackboneWithEBM(nn.Module):
    """ResNet-18 backbone with:
      - Sample-level energy head E_s(x)
      - Feature-level energy head at chosen layer E_f(f_ell)
      - Classifier head

    Returns:
      dict with:
        feats: dict of features
        pooled: (B,D)
        E_s: (B,)
        E_f: (B,)
        s_ch: (B,C_ell) channel scores at feature layer
        logits: (B,num_classes)
    """
    def __init__(self, num_classes: int, feature_layer: str = "layer3"):
        super().__init__()
        base = models.resnet18(weights=None)
        # Expose layers
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool  # outputs (B,512,1,1)

        self.feature_layer = feature_layer
        feat_channels = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}[feature_layer]
        self.sample_head = SampleEnergyHead(in_dim=512)
        self.feature_head = FeatureEnergyHead(in_channels=feat_channels)
        self.classifier = ClassifierHead(in_dim=512, num_classes=num_classes)
        # energy normalizer (EMA)
        self.energy_norm = EnergyNormalizer(
            momentum=cfg.energy_norm_momentum, eps=cfg.energy_norm_eps
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.size(0)
        out = {}
        # Backbone forward
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1p = self.maxpool(x1)

        l1 = self.layer1(x1p)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        feats = {"layer1": l1, "layer2": l2, "layer3": l3, "layer4": l4}
        out["feats"] = feats

        pooled = self.avgpool(l4).view(B, -1)  # (B,512)
        out["pooled"] = pooled

        # Sample-level energy
        E_s = self.sample_head(pooled)  # (B,)
        out["E_s"] = E_s

        # Feature-level energy on chosen layer
        f = feats[self.feature_layer]  # (B,C,H,W)
        E_f, s_ch = self.feature_head(f)
        out["E_f"] = E_f
        out["s_ch"] = s_ch  # (B,C)

        # Classifier logits (no mask applied here; masking is done outside if needed)
        logits = self.classifier(pooled)  # (B,num_classes)
        out["logits"] = logits
        return out
    

    # utility to assemble combined energy E_total
    def combined_energy_from_out(
        self, out: Dict[str, torch.Tensor],
        y: Optional[torch.Tensor] = None,
        alpha: float = 0.5,
        update_norm: bool = True,
        cls_mode: str = "lsep",
        cls_tau: float = 1.0,
    ) -> torch.Tensor:
        """
        E_total = α * E_norm + (1-α) * E_cls
        - E_norm: standardized E_s using EMA (grad flows to sample_head; not to backbone if its params frozen)
        - E_cls:  boundary-aware energy from logits (detached; no grad to backbone/classifier)
        """
        E_s   = out["E_s"]              # (B,)
        logits = out["logits"]           # (B,C)
        E_norm = self.energy_norm(E_s, update=update_norm)    # (B,)
        E_cls  = boundary_energy_from_logits(logits, y, mode=cls_mode, tau=cls_tau)  # (B,)
        return alpha * E_norm + (1.0 - alpha) * E_cls




























