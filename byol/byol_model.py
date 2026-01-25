# byol_model.py

import math
from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, Bottleneck


# -------------------------------
# ResNet-50x2 backbone (<100M params)
# -------------------------------

class ResNet50x2(ResNet):
    """
    ResNet-50 with 2x width (width_per_group=128).
    This matches the "ResNet-50 (2x)" setup from the BYOL paper.
    """
    def __init__(self, **kwargs):
        super().__init__(block=Bottleneck, layers=[3, 4, 6, 3],
                         width_per_group=128, **kwargs)


def build_resnet50x2() -> Tuple[nn.Module, int]:
    """
    Build a ResNet-50x2 backbone that outputs features (no classification head).

    Returns:
        backbone (nn.Module): encoder network
        feat_dim (int): dimensionality of the output features
    """
    backbone = ResNet50x2()
    feat_dim = backbone.fc.in_features  # typically 2048 or 4096 depending on width
    backbone.fc = nn.Identity()         # remove classification head
    return backbone, feat_dim


# -------------------------------
# MLP heads (projector / predictor)
# -------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------------
# BYOL module (online + target)
# -------------------------------

class BYOL(nn.Module):
    """
    BYOL: Bootstrap Your Own Latent (simplified PyTorch version)

    This module assumes you pass TWO views (augmented images) in the forward pass.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        proj_hidden_dim: int = 4096,
        proj_out_dim: int = 256,
    ):
        super().__init__()

        # Online encoder (backbone + projector)
        self.online_encoder = backbone
        self.online_projector = MLP(feat_dim, proj_hidden_dim, proj_out_dim)
        self.online_predictor = MLP(proj_out_dim, proj_hidden_dim, proj_out_dim)

        # Target encoder (no grad, EMA updated)
        self.target_encoder = deepcopy(self.online_encoder)
        self.target_projector = deepcopy(self.online_projector)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _target_forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.target_encoder(x)
        z = self.target_projector(y)
        return z

    def _online_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.online_encoder(x)
        z = self.online_projector(y)
        p = self.online_predictor(z)
        return z, p

    @staticmethod
    def _cosine_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        BYOL loss: || normalize(p) - normalize(z) ||^2 = 2 - 2 * cos_sim(p, z)
        """
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        return 2 - 2 * (p * z).sum(dim=-1)

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        """
        view1, view2: two augmented views of the same batch of images.
        Returns scalar loss.
        """
        # Online network
        z1_online, p1 = self._online_forward(view1)
        z2_online, p2 = self._online_forward(view2)

        # Target network (no grad)
        with torch.no_grad():
            z1_target = self._target_forward(view1)
            z2_target = self._target_forward(view2)

        # Symmetric loss
        loss1 = self._cosine_loss(p1, z2_target)  # view1 predicts target(view2)
        loss2 = self._cosine_loss(p2, z1_target)  # view2 predicts target(view1)
        loss = (loss1 + loss2).mean()
        return loss

    @torch.no_grad()
    def update_moving_average(self, tau: float):
        """
        EMA update for target network parameters:
        target = tau * target + (1 - tau) * online
        """
        for online_params, target_params in [
            (self.online_encoder.parameters(), self.target_encoder.parameters()),
            (self.online_projector.parameters(), self.target_projector.parameters()),
        ]:
            for p_o, p_t in zip(online_params, target_params):
                p_t.data.mul_(tau).add_(p_o.data, alpha=(1.0 - tau))
