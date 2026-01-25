#!/usr/bin/env python3
"""
train_kd96_multicrop_vicreg.py

Feature distillation (KD) at fixed 96x96 resolution with multi-crop:
- Teacher: pretrained DINOv2 ViT-B/14 (frozen), loaded via torch.hub
- Student: randomly initialized ViT-B/14 from dinov2 codebase (<100M),
           trained to match teacher CLS features
- Augmentations: multi-crop at 96x96
    * 2 "global" crops (scale ~0.5–1.0 of original image)
    * V-2 "local" crops (scale ~0.3–0.7 of original image)
    * All resized to 96x96 then padded to 98x98 for ViT-B/14
- Loss: VICReg-style KD
    * For each image:
        - teacher_global[0] is target for student_global[1] + all locals
        - teacher_global[1] is target for student_global[0] + all locals

Run (from dinov2 repo root):
  PYTHONPATH=. torchrun --nproc_per_node=2 train_kd96_multicrop_vicreg.py \
    --data_root /path/to/pretrain \
    --out_dir   /path/to/ckpts \
    --epochs 50 --batch_size 256 \
    --num_views 8
"""

import os
import math
import json
import time
import argparse
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import ConcatDataset
from torchvision import transforms as T
from PIL import Image

from dinov2.models.vision_transformer import vit_base


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ----------------------------
# DDP helpers
# ----------------------------
def ddp_is_on() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def ddp_init():
    """Initialize torch.distributed from torchrun env vars."""
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank


def is_main(rank: int) -> bool:
    return rank == 0


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


# ----------------------------
# Data
# ----------------------------
class UnlabeledImageFolder(Dataset):
    """Recursively load images from a root folder (no labels)."""
    def __init__(self, root: str, transform):
        self.root = root
        self.transform = transform
        self.paths: List[str] = []
        for dp, _, fnames in os.walk(root):
            for f in fnames:
                if os.path.splitext(f)[1].lower() in IMG_EXTS:
                    self.paths.append(os.path.join(dp, f))
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found under {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img)


class MultiCrop96:
    """
    Multi-crop 96x96:
      - num_global crops with larger scale (more context)
      - num_local crops with smaller scale (but not too tiny)
    Output per image: Tensor of shape (V, C, 96, 96)
      where first num_global are "global", rest are "local".
    """
    def __init__(
        self,
        num_views: int,
        num_global: int = 2,
        global_scale=(0.5, 1.0),
        local_scale=(0.3, 0.7),
    ):
        assert num_views >= num_global >= 1
        self.num_global = num_global
        self.num_local = num_views - num_global

        normalize = T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

        base_augs = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply(
                [T.GaussianBlur(kernel_size=7, sigma=(0.1, 1.0))],
                p=0.3,
            ),
        ]

        self.global_aug = T.Compose(
            [
                T.RandomResizedCrop(
                    96,
                    scale=global_scale,
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
                *base_augs,
                T.Pad(padding=1, fill=0, padding_mode="constant"),  # 96 -> 98
                T.ToTensor(),
                normalize,
            ]
        )

        # locals are smaller but not microscopic
        self.local_aug = T.Compose(
            [
                T.RandomResizedCrop(
                    96,
                    scale=local_scale,
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
                *base_augs,
                T.Pad(padding=1, fill=0, padding_mode="constant"),  # 96 -> 98
                T.ToTensor(),
                normalize,
            ]
        )

    def __call__(self, img):
        crops = []
        # global crops
        for _ in range(self.num_global):
            crops.append(self.global_aug(img))
        # local crops
        for _ in range(self.num_local):
            crops.append(self.local_aug(img))
        return torch.stack(crops, dim=0)  # (V, C, 96, 96)


# ----------------------------
# Model utils
# ----------------------------
@torch.no_grad()
def freeze_(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False


def student_cls(student_backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    dinov2 ViT returns a dict with x_norm_clstoken when is_training=True.
    """
    out = student_backbone(x, is_training=True)
    return out["x_norm_clstoken"]  # (B, 768)


# ----------------------------
# VICReg-KD loss
# ----------------------------
def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """
    Return a flat view of the off-diagonal elements of a square matrix.
    x: (D, D)
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_kd_loss(
    z_s: torch.Tensor,
    z_t: torch.Tensor,
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    eps: float = 1e-4,
    gamma: float = 1.0,
):
    """
    VICReg-style loss adapted for teacher-student KD:

      - Invariance: MSE(z_s, z_t)
      - Variance: variance regularization on student only
      - Covariance: decorrelation on student only

    z_s: (N, D) student embeddings
    z_t: (N, D) teacher embeddings (no grad)
    """
    # make sure teacher is detached
    z_t = z_t.detach()

    # 1) Invariance loss (MSE between student and teacher)
    sim_loss = F.mse_loss(z_s, z_t)

    # 2) Variance loss (student only)
    std = torch.sqrt(z_s.var(dim=0, unbiased=False) + eps)  # (D,)
    var_loss = F.relu(gamma - std).mean()

    # 3) Covariance loss (student only)
    N, D = z_s.shape
    assert N > 1, "Batch too small for covariance term (N must be > 1)."
    z_centered = z_s - z_s.mean(dim=0, keepdim=True)
    cov = (z_centered.T @ z_centered) / (N - 1)  # (D, D)
    cov_loss = _off_diagonal(cov).pow(2).sum() / D

    loss = (
        sim_coeff * sim_loss
        + std_coeff * var_loss
        + cov_coeff * cov_loss
    )

    stats = {
        "sim_loss": sim_loss.detach(),
        "var_loss": var_loss.detach(),
        "cov_loss": cov_loss.detach(),
        "total_loss": loss.detach(),
    }
    return loss, stats


def save_ckpt(path: str, payload: dict):
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_root",
        action="append",
        required=True,
        help="Path to an unlabeled image folder. Pass this flag multiple times for multiple datasets.",
    )
    ap.add_argument("--out_dir", required=True, help="Where to write checkpoints")
    ap.add_argument("--teacher_repo", default="facebookresearch/dinov2")
    ap.add_argument("--teacher_name", default="dinov2_vitb14")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument(
        "--num_views",
        type=int,
        default=8,
        help="How many augmented views per image (all 96x96). "
             "First 2 are global, rest are local.",
    )
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument(
        "--patch",
        type=int,
        default=16,
        help="Student patch size (96 must be divisible by patch)",
    )
    ap.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint (ckpt_epoch_XXX.pth) to resume from",
    )

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # DDP setup
    if ddp_is_on():
        rank, world, local_rank = ddp_init()
        device = torch.device("cuda", local_rank)
    else:
        rank, world, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Save config (rank0)
    if is_main(rank):
        with open(os.path.join(args.out_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    # 1) Data: Multi-crop 96x96
    tfm = MultiCrop96(num_views=args.num_views)

    datasets = [UnlabeledImageFolder(r, transform=tfm) for r in args.data_root]
    if is_main(rank):
        for r, d in zip(args.data_root, datasets):
            print(f"[data] {r}: {len(d)} images")
        print(f"[data] total: {sum(len(d) for d in datasets)} images")

    ds = ConcatDataset(datasets)
    sampler = DistributedSampler(ds, shuffle=True) if ddp_is_on() else None

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    # 2) Teacher
    if ddp_is_on() and is_main(rank):
        _ = torch.hub.load(args.teacher_repo, args.teacher_name)
    barrier()

    teacher = torch.hub.load(args.teacher_repo, args.teacher_name)
    teacher = teacher.to(device).eval()
    freeze_(teacher)

    # 3) Student + Opt + Scaler
    student = vit_base(patch_size=14, img_size=98, num_register_tokens=0).to(device)
    student.train()

    if ddp_is_on():
        student = nn.parallel.DistributedDataParallel(
            student,
            device_ids=[local_rank],
            find_unused_parameters=False,
        )

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # 4) Resume logic
    start_epoch = 0
    global_step = 0

    if args.resume is not None:
        map_loc = device if device.type == "cuda" else "cpu"
        if is_main(rank):
            print(f"[resume] Loading checkpoint from {args.resume}")
        ckpt = torch.load(args.resume, map_location=map_loc)

        state_dict = ckpt["student_state"]
        if ddp_is_on():
            student.module.load_state_dict(state_dict)
        else:
            student.load_state_dict(state_dict)

        opt.load_state_dict(ckpt["opt_state"])
        scaler.load_state_dict(ckpt["scaler_state"])

        ckpt_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt_epoch + 1

        if is_main(rank):
            print(
                f"[resume] Resumed from epoch {ckpt_epoch} "
                f"(next epoch = {start_epoch}), global_step={global_step}"
            )

    # 5) LR schedule
    steps_per_epoch = len(dl)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return args.lr * (step + 1) / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * args.lr * (1.0 + math.cos(math.pi * t))

    start_time = time.time()

    # 6) Training loop
    for epoch in range(start_epoch, args.epochs):
        if ddp_is_on():
            sampler.set_epoch(epoch)

        for it, views in enumerate(dl):
            views = views.to(device, non_blocking=True)   # (B, V, C, H, W)
            B, V, C, H, W = views.shape

            num_global = 2
            assert V >= num_global, "num_views must be >= 2 for multi-crop."
            num_local = V - num_global

            # Split global / local views
            global_views = views[:, :num_global]              # (B, 2, C, H, W)

            # Flatten globals for teacher
            # global_flat = global_views.view(B * num_global, C, H, W)  # (B*2, C, H, W)
            global_flat = global_views.reshape(B * num_global, C, H, W)

            # ---- LR update ----
            lr = lr_at(global_step)
            for pg in opt.param_groups:
                pg["lr"] = lr

            # ---- TEACHER: only on global crops ----
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                z_t_global = teacher(global_flat)             # (B*2, D)
                D = z_t_global.size(-1)
                z_t_global = z_t_global.view(B, num_global, D)  # (B, 2, D)

                # teacher globals per image
                t0 = z_t_global[:, 0, :]   # (B, D)
                t1 = z_t_global[:, 1, :]   # (B, D)

            # ---- STUDENT: on ALL crops (global + local) ----
            # x_all = views.view(B * V, C, H, W)               # (B*V, C, H, W)
            x_all = views.reshape(B * V, C, H, W)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                s_model = student.module if ddp_is_on() else student
                z_s_all = student_cls(s_model, x_all)        # (B*V, D)
                z_s_all = z_s_all.view(B, V, D)              # (B, V, D)

                # Split student features into globals + locals
                z_s_global = z_s_all[:, :num_global, :]      # (B, 2, D)
                z_s_local = z_s_all[:, num_global:, :]       # (B, L, D) (L may be 0)

                L_local = z_s_local.size(1)

                # Build student sets:
                # For teacher t0: student_global[1] + all locals
                s_to_t0 = torch.cat(
                    [z_s_global[:, 1:2, :], z_s_local], dim=1
                )  # (B, 1+L, D)

                # For teacher t1: student_global[0] + all locals
                s_to_t1 = torch.cat(
                    [z_s_global[:, 0:1, :], z_s_local], dim=1
                )  # (B, 1+L, D)

                # Flatten for VICReg-KD
                s_to_t0_flat = s_to_t0.reshape(-1, D)        # (B*(1+L), D)
                s_to_t1_flat = s_to_t1.reshape(-1, D)        # (B*(1+L), D)

                # Repeat teachers to match
                k = 1 + L_local
                t0_rep = t0.unsqueeze(1).expand(-1, k, -1).reshape(-1, D)
                t1_rep = t1.unsqueeze(1).expand(-1, k, -1).reshape(-1, D)

                # VICReg-KD for both teacher views, then average
                loss0, stats0 = vicreg_kd_loss(
                    s_to_t0_flat,
                    t0_rep,
                    sim_coeff=25.0,
                    std_coeff=25.0,
                    cov_coeff=1.0,
                )
                loss1, stats1 = vicreg_kd_loss(
                    s_to_t1_flat,
                    t1_rep,
                    sim_coeff=25.0,
                    std_coeff=25.0,
                    cov_coeff=1.0,
                )

                loss = 0.5 * (loss0 + loss1)

                # average stats for logging
                loss_stats = {
                    "sim_loss": 0.5 * (stats0["sim_loss"] + stats1["sim_loss"]),
                    "var_loss": 0.5 * (stats0["var_loss"] + stats1["var_loss"]),
                    "cov_loss": 0.5 * (stats0["cov_loss"] + stats1["cov_loss"]),
                    "total_loss": loss.detach(),
                }

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)

            scaler.step(opt)
            scaler.update()

            if is_main(rank) and (global_step % args.log_every == 0):
                elapsed = time.time() - start_time
                imgs = (global_step + 1) * args.batch_size * (
                    dist.get_world_size() if ddp_is_on() else 1
                )
                ips = imgs / max(1e-6, elapsed)
                print(
                    f"[ep {epoch:03d} it {it:05d} step {global_step:06d}] "
                    f"loss={loss.item():.4f} "
                    f"sim={loss_stats['sim_loss']:.4f} "
                    f"var={loss_stats['var_loss']:.4f} "
                    f"cov={loss_stats['cov_loss']:.4f} "
                    f"lr={lr:.2e} imgs/s={ips:.1f}"
                )

            global_step += 1

        if is_main(rank):
            state = student.module.state_dict() if ddp_is_on() else student.state_dict()
            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "student_state": state,
                "opt_state": opt.state_dict(),
                "scaler_state": scaler.state_dict(),
                "args": vars(args),
            }
            save_ckpt(os.path.join(args.out_dir, f"ckpt_epoch_{epoch:03d}.pth"), ckpt)

    if is_main(rank):
        state = student.module.state_dict() if ddp_is_on() else student.state_dict()
        torch.save(state, os.path.join(args.out_dir, "student_backbone_only.pth"))

    if ddp_is_on():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
