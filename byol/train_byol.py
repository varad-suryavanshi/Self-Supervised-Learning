# train_byol.py

import argparse
import math
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from byol_model import build_resnet50x2, BYOL
from datasets import BYOLPretrainDataset, get_byol_transforms


# -------------------------------
# LARS optimizer
# -------------------------------

class LARS(torch.optim.Optimizer):
    """
    Layer-wise Adaptive Rate Scaling (LARS) optimizer.
    Suitable for large-batch training as used in BYOL / SimCLR.

    This is SGD + momentum + weight decay, with layer-wise adaptive learning rate.
    """

    def __init__(
        self,
        params,
        lr=0.1,
        weight_decay=1.5e-6,
        momentum=0.9,
        eta=0.001,
        eps=1e-9,
        exclude_bias_and_norm=True,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            eps=eps,
            exclude_bias_and_norm=exclude_bias_and_norm,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eta = group["eta"]
            eps = group["eps"]
            exclude_bias_and_norm = group["exclude_bias_and_norm"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("LARS does not support sparse gradients")

                # Optionally exclude bias and norm params from weight decay and LARS
                if exclude_bias_and_norm and (p.ndim == 1):
                    # 1D params are usually bias or norm weight
                    grad = grad
                    param_norm = None
                    grad_norm = None
                    trust_ratio = 1.0
                    wd = 0.0
                else:
                    wd = weight_decay
                    grad = grad + wd * p
                    param_norm = torch.norm(p)
                    grad_norm = torch.norm(grad)
                    trust_ratio = 1.0
                    if param_norm > 0 and grad_norm > 0:
                        trust_ratio = eta * param_norm / (grad_norm + eps)

                # Momentum
                state = self.state[p]
                if "momentum_buffer" not in state:
                    buf = state["momentum_buffer"] = torch.zeros_like(p)
                else:
                    buf = state["momentum_buffer"]

                buf.mul_(momentum).add_(grad, alpha=trust_ratio * lr)
                p.add_(-buf)

        return loss


# -------------------------------
# Schedules
# -------------------------------

def cosine_lr_schedule(global_step, total_steps, base_lr, warmup_steps):
    if global_step < warmup_steps:
        return base_lr * float(global_step) / float(max(1, warmup_steps))
    progress = float(global_step - warmup_steps) / float(
        max(1, total_steps - warmup_steps)
    )
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def ema_tau_schedule(global_step, total_steps, tau_base):
    """
    Cosine schedule for EMA coefficient, as in BYOL:
    tau_k = 1 - (1 - tau_base) * (cos(pi k / K) + 1) / 2
    """
    frac = float(global_step) / float(max(1, total_steps))
    return 1.0 - (1.0 - tau_base) * 0.5 * (1.0 + math.cos(math.pi * frac))


# -------------------------------
# Distributed setup
# -------------------------------

def setup_distributed(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# -------------------------------
# Training loop
# -------------------------------

def train(rank, world_size, args):
    setup_distributed(rank, world_size)

    is_main = (rank == 0)

    # Build model
    backbone, feat_dim = build_resnet50x2()
    backbone.cuda(rank)
    model = BYOL(
        backbone=backbone,
        feat_dim=feat_dim,
        proj_hidden_dim=args.proj_hidden_dim,
        proj_out_dim=args.proj_out_dim,
    )
    model.cuda(rank)

    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # Data
    # t1, t2 = get_byol_transforms(image_size=args.image_size)
    # dataset = BYOLPretrainDataset(args.data_dir, t1, t2)
    # sampler = DistributedSampler(dataset)
    # loader = DataLoader(
    #     dataset,
    #     batch_size=args.batch_size_per_gpu,
    #     sampler=sampler,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     drop_last=True,
    # )

    t1, t2 = get_byol_transforms(image_size=args.image_size)

    root_dirs = [
        "/scratch/vs3273/DL_pretrain_500k/train",
        "/scratch/vs3273/DL_pretrain_1.5M/pass_96",
        "/scratch/vs3273/DL_pretrain_1.5M/openimages_96",
    ]

    # dataset = BYOLPretrainDataset(root_dirs, t1, t2)
    cache_list_path = "/scratch/vs3273/DL_pretrain_1.5M/filelist_1p05M.txt"

    dataset = BYOLPretrainDataset(
        root_dirs=root_dirs,
        transform1=t1,
        transform2=t2,
        cache_list_path=cache_list_path,
    )
    sampler = DistributedSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size_per_gpu,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)

    # Optimizer (LARS)
    base_lr = args.base_lr  # already scaled for global batch
    optimizer = LARS(
        model.parameters(),
        lr=base_lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        eta=args.lars_eta,
        exclude_bias_and_norm=True,
    )

    global_step = 0

    if is_main:
        print(f"World size: {world_size}")
        print(f"Dataset size: {len(dataset)} images")
        print(f"Steps/epoch: {steps_per_epoch}, total steps: {total_steps}")
        print(f"Base LR: {base_lr}")
        print(f"Tau base: {args.tau_base}")

    model.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for step, (v1, v2) in enumerate(loader):
            v1 = v1.cuda(rank, non_blocking=True)
            v2 = v2.cuda(rank, non_blocking=True)

            # LR schedule
            lr = cosine_lr_schedule(global_step, total_steps, base_lr, warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Forward
            loss = model(v1, v2)
            loss = loss / args.grad_accum_steps
            loss.backward()

            # Gradient accumulation
            if (global_step + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                # EMA for target network
                tau = ema_tau_schedule(global_step, total_steps, args.tau_base)
                model.module.update_moving_average(tau)

            # For logging
            epoch_loss += loss.item() * args.grad_accum_steps
            num_batches += 1
            global_step += 1

        # Average loss across processes
        epoch_loss_tensor = torch.tensor(epoch_loss / num_batches, device=rank)
        dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.AVG)
        epoch_loss_avg = epoch_loss_tensor.item()

        if is_main:
            elapsed = time.time() - start_time
            print(
                f"Epoch [{epoch+1}/{args.epochs}] "
                f"Loss: {epoch_loss_avg:.4f} "
                f"LR: {lr:.6f} "
                f"Time: {elapsed/60:.1f} min"
            )

        # Save checkpoint from main process
        if is_main and ((epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs):
            ckpt = {
                "epoch": epoch + 1,
                "model_state": model.module.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "global_step": global_step,
                "steps_per_epoch": steps_per_epoch,
            }
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_path = os.path.join(args.output_dir, f"byol_epoch_{epoch+1}.pth")
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    cleanup_distributed()


# -------------------------------
# Entry point
# -------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="BYOL pretraining (ResNet-50x2 + LARS)")

    parser.add_argument("--data-dir", type=str, required=False,
                        help="Path to pretrain/ directory (unlabeled images)")
    parser.add_argument("--output-dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")

    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--batch-size-per-gpu", type=int, default=256,
                        help="Per-GPU batch size. Global batch = this * world_size.")
    parser.add_argument("--epochs", type=int, default=640)
    parser.add_argument("--warmup-epochs", type=float, default=10.0)

    # BYOL head dims
    parser.add_argument("--proj-hidden-dim", type=int, default=4096)
    parser.add_argument("--proj-out-dim", type=int, default=256)

    # Optimizer (LARS) hyperparams
    parser.add_argument("--base-lr", type=float, default=0.8,
                        help="Base LR (already scaled for global batch ≈ 1024).")
    parser.add_argument("--weight-decay", type=float, default=1.5e-6)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lars-eta", type=float, default=0.001)

    # EMA
    parser.add_argument("--tau-base", type=float, default=0.996)

    # Training mechanics
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Gradient accumulation steps. 1 means no accumulation.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save checkpoint every N epochs")

    return parser.parse_args()


def main():
    args = parse_args()

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("Need at least 1 GPU to run BYOL training")

    mp.spawn(
        train,
        args=(world_size, args),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
