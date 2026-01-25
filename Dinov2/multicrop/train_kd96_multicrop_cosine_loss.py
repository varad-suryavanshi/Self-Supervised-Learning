#!/usr/bin/env python3
"""
train_kd96_multicrop_cosine_loss.py

Feature distillation (KD) at fixed 96x96 resolution:
- Teacher: pretrained DINOv2 ViT-B/14 (frozen), loaded via torch.hub
- Student: randomly initialized ViT-B/14 from dinov2 codebase (<100M), trained to match teacher CLS features
- Loss: cosine distance between normalized features

Run (from dinov2 repo root):
  PYTHONPATH=. torchrun --nproc_per_node=2 train_kd96.py \
    --data_root /path/to/pretrain \
    --out_dir   /path/to/ckpts \
    --epochs 50 --batch_size 256
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
from torchvision import transforms as T
from PIL import Image
from torch.utils.data import ConcatDataset  # add near imports

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


# class MultiView96:
#     """
#     Return N independently augmented views, each strictly 96x96.
#     Output: Tensor of shape (V, C, 96, 96)
#     """
#     def __init__(self, num_views: int):
#         self.num_views = num_views
#         normalize = T.Normalize(mean=(0.485, 0.456, 0.406),
#                                 std=(0.229, 0.224, 0.225))
#         # IMPORTANT: output is ALWAYS 96x96
#         self.aug = T.Compose([
#             T.RandomResizedCrop(96, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
#             T.RandomHorizontalFlip(p=0.5),
#             T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
#             T.RandomGrayscale(p=0.2),
#             T.RandomApply([T.GaussianBlur(kernel_size=7, sigma=(0.1, 1.0))], p=0.3),

#             T.Pad(padding=1, fill=0, padding_mode="constant"),  # 96 -> 98

#             T.ToTensor(),
#             normalize,
#         ])


#     def __call__(self, img):
#         views = [self.aug(img) for _ in range(self.num_views)]
#         return torch.stack(views, dim=0)  # (V, C, 96, 96)


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
            T.RandomApply([T.GaussianBlur(kernel_size=7, sigma=(0.1, 1.0))], p=0.3),
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

        # NOTE: local_scale=(0.3, 0.7) so locals are smaller, but not microscopic.
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
        # First num_global = "global" crops
        for _ in range(self.num_global):
            crops.append(self.global_aug(img))
        # Then num_local = "local" crops (if any)
        for _ in range(self.num_local):
            crops.append(self.local_aug(img))
        return torch.stack(crops, dim=0)  # (V, C, 96, 96) with V = num_views



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
# Training
# ----------------------------
def cosine_dist(z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
    """
    z_s, z_t normalized => dot product is cosine similarity.
    Return mean cosine distance.
    """
    z_s = F.normalize(z_s, dim=-1)
    z_t = F.normalize(z_t, dim=-1)
    cos = (z_s * z_t).sum(dim=-1)  # (B,)
    return (1.0 - cos).mean()


# def multi_crop_cosine_kd(t_global, s_global, s_local):
#     """
#     SwAV-style multi-crop KD with cosine distance.

#     t_global: (B, G, D) teacher feats for G global crops (G=2)
#     s_global: (B, G, D) student feats for G global crops
#     s_local:  (B, L, D) student feats for L local crops (maybe 0)

#     For each image:
#       - teacher[0] is matched by student global[1] + all locals
#       - teacher[1] is matched by student global[0] + all locals
#     """
#     B, G, D = s_global.shape
#     assert G == 2, "This implementation assumes 2 global crops."
#     L = s_local.size(1) if s_local.numel() > 0 else 0

#     # Normalize for cosine
#     t_global = F.normalize(t_global, dim=-1)   # (B,2,D)
#     s_global = F.normalize(s_global, dim=-1)   # (B,2,D)
#     if L > 0:
#         s_local = F.normalize(s_local, dim=-1)     # (B,L,D)

#     # teacher globals
#     t0 = t_global[:, 0, :]    # (B, D)
#     t1 = t_global[:, 1, :]    # (B, D)

#     # students that should match t0: global[1] + locals
#     if L > 0:
#         s_to_t0 = torch.cat([s_global[:, 1:2, :], s_local], dim=1)  # (B, 1+L, D)
#     else:
#         s_to_t0 = s_global[:, 1:2, :]  # (B,1,D)

#     # students that should match t1: global[0] + locals
#     if L > 0:
#         s_to_t1 = torch.cat([s_global[:, 0:2-1, :], s_local], dim=1)  # (B, 1+L, D)
#         # (the [:0:1] trick is just to keep shapes; could also write [:,0:1,:])
#         s_to_t1 = torch.cat([s_global[:, 0:1, :], s_local], dim=1)
#     else:
#         s_to_t1 = s_global[:, 0:1, :]  # (B,1,D)

#     # Broadcast teachers over the (1+L) views
#     def pairwise_cosine_loss(s_views, t_vec):
#         """
#         s_views: (B, K, D)
#         t_vec:   (B, D)
#         matches each of the K views to the same teacher vector
#         """
#         B, K, D = s_views.shape
#         t_exp = t_vec.unsqueeze(1).expand(-1, K, -1)  # (B,K,D)
#         cos = (s_views * t_exp).sum(dim=-1)           # (B,K)
#         # cosine distance = 1 - cos
#         return (1.0 - cos).mean()

#     loss0 = pairwise_cosine_loss(s_to_t0, t0)
#     loss1 = pairwise_cosine_loss(s_to_t1, t1)

#     return 0.5 * (loss0 + loss1)


def multi_crop_cosine_kd(t_global, s_global, s_local):
    B, G, D = s_global.shape
    assert G == 2, "This implementation assumes 2 global crops."
    L = s_local.size(1) if s_local.numel() > 0 else 0

    # Normalize for cosine
    t_global = F.normalize(t_global, dim=-1)   # (B,2,D)
    s_global = F.normalize(s_global, dim=-1)   # (B,2,D)
    if L > 0:
        s_local = F.normalize(s_local, dim=-1)  # (B,L,D)

    # teacher globals
    t0 = t_global[:, 0, :]    # (B, D)
    t1 = t_global[:, 1, :]    # (B, D)

    # students that should match t0: global[1] + locals
    if L > 0:
        s_to_t0 = torch.cat([s_global[:, 1:2, :], s_local], dim=1)  # (B, 1+L, D)
    else:
        s_to_t0 = s_global[:, 1:2, :]  # (B,1,D)

    # students that should match t1: global[0] + locals
    if L > 0:
        s_to_t1 = torch.cat([s_global[:, 0:1, :], s_local], dim=1)  # (B, 1+L, D)
    else:
        s_to_t1 = s_global[:, 0:1, :]  # (B,1,D)

    def pairwise_cosine_loss(s_views, t_vec):
        """
        s_views: (B, K, D)
        t_vec:   (B, D)
        matches each of the K views to the same teacher vector
        """
        B, K, D = s_views.shape
        t_exp = t_vec.unsqueeze(1).expand(-1, K, -1)  # (B,K,D)
        cos = (s_views * t_exp).sum(dim=-1)           # (B,K)
        return (1.0 - cos).mean()                     # cosine distance

    loss0 = pairwise_cosine_loss(s_to_t0, t0)
    loss1 = pairwise_cosine_loss(s_to_t1, t1)

    return 0.5 * (loss0 + loss1)




def save_ckpt(path: str, payload: dict):
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--data_root", action="append", required=True, help="Path to an unlabeled image folder. Pass this flag multiple times for multiple datasets.")
#     ap.add_argument("--out_dir", required=True, help="Where to write checkpoints")
#     ap.add_argument("--teacher_repo", default="facebookresearch/dinov2")
#     ap.add_argument("--teacher_name", default="dinov2_vitb14")
#     ap.add_argument("--epochs", type=int, default=50)
#     ap.add_argument("--batch_size", type=int, default=256)
#     ap.add_argument("--num_views", type=int, default=2, help="How many augmented views per image (all 96x96)")
#     ap.add_argument("--lr", type=float, default=5e-4)
#     ap.add_argument("--wd", type=float, default=0.05)
#     ap.add_argument("--warmup_epochs", type=int, default=2)
#     ap.add_argument("--grad_clip", type=float, default=1.0)
#     ap.add_argument("--num_workers", type=int, default=8)
#     ap.add_argument("--log_every", type=int, default=50)
#     ap.add_argument("--patch", type=int, default=16, help="Student patch size (96 must be divisible by patch)")

#     args = ap.parse_args()


#     os.makedirs(args.out_dir, exist_ok=True)

#     # DDP setup (works in single-GPU too)
#     if ddp_is_on():
#         rank, world, local_rank = ddp_init()
#         device = torch.device("cuda", local_rank)
#     else:
#         rank, world, local_rank = 0, 1, 0
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Save config (rank0)
#     if is_main(rank):
#         with open(os.path.join(args.out_dir, "config.json"), "w") as f:
#             json.dump(vars(args), f, indent=2)

#     # 1) Data
#     tfm = MultiView96(num_views=args.num_views)


#     datasets = [UnlabeledImageFolder(r, transform=tfm) for r in args.data_root]
#     if is_main(rank):
#         for r, d in zip(args.data_root, datasets):
#             print(f"[data] {r}: {len(d)} images")
#         print(f"[data] total: {sum(len(d) for d in datasets)} images")

#     ds = ConcatDataset(datasets)
#     sampler = DistributedSampler(ds, shuffle=True) if ddp_is_on() else None

#     dl = DataLoader(
#         ds,
#         batch_size=args.batch_size,
#         sampler=sampler,
#         shuffle=(sampler is None),
#         num_workers=args.num_workers,
#         pin_memory=True,
#         drop_last=True,
#         persistent_workers=(args.num_workers > 0),
#     )


#     if ddp_is_on() and is_main(rank):
#         _ = torch.hub.load(args.teacher_repo, args.teacher_name)
#     barrier()

#     teacher = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
#     teacher = teacher.to(device).eval()
#     for p in teacher.parameters():
#         p.requires_grad = False 


#     student = vit_base(patch_size=14, img_size=98, num_register_tokens=0).to(device)
#     student.train()

#     if ddp_is_on():
#         student = nn.parallel.DistributedDataParallel(student, device_ids=[local_rank], find_unused_parameters=False)

#     opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.wd)
#     scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

#     steps_per_epoch = len(dl)
#     total_steps = args.epochs * steps_per_epoch
#     warmup_steps = args.warmup_epochs * steps_per_epoch

#     def lr_at(step: int) -> float:
#         if step < warmup_steps:
#             return args.lr * (step + 1) / max(1, warmup_steps)
#         t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
#         return 0.5 * args.lr * (1.0 + math.cos(math.pi * t))

#     global_step = 0
#     start_time = time.time()

#     for epoch in range(args.epochs):
#         if ddp_is_on():
#             sampler.set_epoch(epoch)

#         for it, views in enumerate(dl):
#             views = views.to(device, non_blocking=True)
#             b, v, c, h, w = views.shape
#             x = views.view(b * v, c, h, w)

#             lr = lr_at(global_step)
#             for pg in opt.param_groups:
#                 pg["lr"] = lr

#             with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
#                 z_t = teacher(x)  # (B*V, 768) for vitb14

#             with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
#                 s = student.module if ddp_is_on() else student
#                 z_s = student_cls(s, x)  # (B*V, 768)
#                 loss = cosine_dist(z_s, z_t)

#             opt.zero_grad(set_to_none=True)
#             scaler.scale(loss).backward()

#             if args.grad_clip and args.grad_clip > 0:
#                 scaler.unscale_(opt)
#                 torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)

#             scaler.step(opt)
#             scaler.update()

#             if is_main(rank) and (global_step % args.log_every == 0):
#                 elapsed = time.time() - start_time
#                 imgs = (global_step + 1) * args.batch_size * (dist.get_world_size() if ddp_is_on() else 1)
#                 ips = imgs / max(1e-6, elapsed)
#                 print(f"[ep {epoch:03d} it {it:05d} step {global_step:06d}] "
#                       f"loss={loss.item():.4f} lr={lr:.2e} imgs/s={ips:.1f}")

#             global_step += 1

#         if is_main(rank):
#             state = student.module.state_dict() if ddp_is_on() else student.state_dict()
#             ckpt = {
#                 "epoch": epoch,
#                 "global_step": global_step,
#                 "student_state": state,
#                 "opt_state": opt.state_dict(),
#                 "scaler_state": scaler.state_dict(),
#                 "args": vars(args),
#             }
#             save_ckpt(os.path.join(args.out_dir, f"ckpt_epoch_{epoch:03d}.pth"), ckpt)

#     if is_main(rank):
#         state = student.module.state_dict() if ddp_is_on() else student.state_dict()
#         torch.save(state, os.path.join(args.out_dir, "student_backbone_only.pth"))

#     if ddp_is_on():
#         dist.destroy_process_group()

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
        default=2,
        help="How many augmented views per image (all 96x96)",
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

    # -------------------------
    # DDP setup
    # -------------------------
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

    # -------------------------
    # Data
    # -------------------------
    # tfm = MultiView96(num_views=args.num_views)
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

    # -------------------------
    # Teacher
    # -------------------------
    if ddp_is_on() and is_main(rank):
        _ = torch.hub.load(args.teacher_repo, args.teacher_name)
    barrier()

    teacher = torch.hub.load(args.teacher_repo, args.teacher_name)
    teacher = teacher.to(device).eval()
    freeze_(teacher)

    # -------------------------
    # Student + Opt + Scaler
    # -------------------------
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

    # -------------------------
    # Resume logic
    # -------------------------
    start_epoch = 0
    global_step = 0

    if args.resume is not None:
        map_loc = device if device.type == "cuda" else "cpu"
        if is_main(rank):
            print(f"[resume] Loading checkpoint from {args.resume}")
        ckpt = torch.load(args.resume, map_location=map_loc)

        # Load student weights
        state_dict = ckpt["student_state"]
        if ddp_is_on():
            student.module.load_state_dict(state_dict)
        else:
            student.load_state_dict(state_dict)

        # Load optimizer & scaler
        opt.load_state_dict(ckpt["opt_state"])
        scaler.load_state_dict(ckpt["scaler_state"])

        # Epoch/global step
        ckpt_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)

        # We saved at end of epoch, so resume from next one
        start_epoch = ckpt_epoch + 1

        if is_main(rank):
            print(
                f"[resume] Resumed from epoch {ckpt_epoch} "
                f"(next epoch = {start_epoch}), global_step={global_step}"
            )

    # -------------------------
    # LR schedule
    # -------------------------
    steps_per_epoch = len(dl)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return args.lr * (step + 1) / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * args.lr * (1.0 + math.cos(math.pi * t))

    start_time = time.time()

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(start_epoch, args.epochs):
        if ddp_is_on():
            sampler.set_epoch(epoch)

        for it, views in enumerate(dl):
            # views = views.to(device, non_blocking=True)
            # b, v, c, h, w = views.shape
            # x = views.view(b * v, c, h, w)

            views = views.to(device, non_blocking=True)   # (B, V, C, H, W)
            B, V, C, H, W = views.shape
            num_global = 2
            assert V >= num_global, "num_views must be >= 2 for multi-crop."

            num_local = V - num_global

            # Split into global + local crops
            global_views = views[:, :num_global]                     # (B, 2, C, H, W)
            local_views  = views[:, num_global:] if num_local > 0 else None  # (B, L, C, H, W)

            # Flatten batch for teacher/student
            global_flat = global_views.reshape(B * num_global, C, H, W)      # (B*2, C, H, W)
            if num_local > 0:
                local_flat = local_views.reshape(B * num_local, C, H, W)     # (B*L, C, H, W)

            # ----------------------------
            # Teacher forward on GLOBALS ONLY
            # ----------------------------
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                z_t_global = teacher(global_flat)    # (B*2, 768) for vitb14

            # ----------------------------
            # Student forward on ALL crops
            # ----------------------------
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                s_model = student.module if ddp_is_on() else student

                # student on global crops
                z_s_global = student_cls(s_model, global_flat)      # (B*2, 768)

                if num_local > 0:
                    z_s_local = student_cls(s_model, local_flat)    # (B*L, 768)

                # reshape back to (B, G, D) and (B, L, D)
                D = z_s_global.size(-1)
                z_t_global = z_t_global.view(B, num_global, D)      # teacher globals (B,2,D)
                z_s_global = z_s_global.view(B, num_global, D)      # student globals (B,2,D)

                if num_local > 0:
                    z_s_local = z_s_local.view(B, num_local, D)     # student locals (B,L,D)
                else:
                    # create an empty tensor to keep the API simple
                    z_s_local = torch.empty(B, 0, D, device=z_s_global.device, dtype=z_s_global.dtype)

                # SwAV-style multi-crop KD with cosine distance
                loss = multi_crop_cosine_kd(z_t_global, z_s_global, z_s_local)


            # LR update
            lr = lr_at(global_step)
            for pg in opt.param_groups:
                pg["lr"] = lr

            # Teacher forward (no grad)
            # with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            #     z_t = teacher(x)  # (B*V, 768) for vitb14

            # # Student forward
            # with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            #     s = student.module if ddp_is_on() else student
            #     z_s = student_cls(s, x)  # (B*V, 768)
            #     loss = cosine_dist(z_s, z_t)

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
                    f"loss={loss.item():.4f} lr={lr:.2e} imgs/s={ips:.1f}"
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
