# run_dl_pretrain.py
import argparse
import torch
import torch.backends.cudnn as cudnn

from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
from datasets.dl_pretrain_dataset import DLPretrainDataset


def parse_args():
    parser = argparse.ArgumentParser(description="SimCLR on DL_pretrain 96px")

    # Data
    parser.add_argument(
        "--data-roots",
        type=str,
        default="/vast/vs3273/DL_pretrain_500k/train,"
                "/vast/vs3273/DL_pretrain_1.5M/pass_96,"
                "/vast/vs3273/DL_pretrain_1.5M/openimages_96",
        help="Comma-separated list of dataset roots.",
    )
    parser.add_argument(
        "--filelist-path",
        type=str,
        default="/vast/vs3273/DL_pretrain_1.5M/filelist_simclr_96px.txt",
        help="Path to cached filelist of image paths.",
    )
    parser.add_argument(
        "--image-size", type=int, default=96, help="Input resolution."
    )
    parser.add_argument(
        "--min-scale",
        type=float,
        default=0.2,
        help="Minimum crop area scale for RandomResizedCrop.",
    )

    # Model
    parser.add_argument(
        "-a",
        "--arch",
        default="resnet50",
        choices=["resnet18", "resnet50"],
        help="Backbone architecture.",
    )
    parser.add_argument(
        "--out_dim",
        default=128,
        type=int,
        help="Projection head output dimension.",
    )

    # Training
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument(
        "-b",
        "--batch-size",
        default=1024,
        type=int,
        help="Global batch size (DataParallel splits this across GPUs).",
    )
    parser.add_argument(
        "--effective-batch-size",
        default=4096,
        type=int,
        help="Effective batch size for LR scaling via grad accumulation.",
    )
    parser.add_argument(
        "--workers",
        default=12,
        type=int,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--temperature",
        default=0.1,
        type=float,
        help="Softmax temperature.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        default=100,
        type=int,
        help="Log every n steps.",
    )
    parser.add_argument(
        "--fp16-precision",
        action="store_true",
        help="Use 16-bit precision training.",
    )

    parser.add_argument(
        "--ckpt-freq",
        type=int,
        default=10,   # save every 10 epochs; set 0 to only save at end
        help="Save checkpoint every N epochs (0 = only at the end).",
    )


    # Misc
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--disable-cuda",
        action="store_true",
        help="Force CPU (debug only).",
    )

    args = parser.parse_args()

    # Device
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device("cpu")

    # n_views fixed to 2 for SimCLR
    args.n_views = 2

    # Grad accumulation / effective batch
    if args.effective_batch_size is not None:
        assert (
            args.effective_batch_size % args.batch_size == 0
        ), "effective-batch-size must be multiple of batch-size"
        args.accum_steps = args.effective_batch_size // args.batch_size
    else:
        args.accum_steps = 1

    # LR scaling like SimCLR: lr = 0.3 * (B_eff / 256)
    B_eff = args.effective_batch_size or args.batch_size
    args.lr = 0.3 * (B_eff / 256.0)
    print(
        f"Using lr={args.lr:.5f} for effective batch size {B_eff} "
        f"(batch_size={args.batch_size}, accum_steps={args.accum_steps})"
    )

    return args


def main():
    args = parse_args()

    # Parse data roots
    roots = [r.strip() for r in args.data_roots.split(",") if r.strip()]
    print("Data roots:")
    for r in roots:
        print("  ", r)

    # Dataset & DataLoader
    train_dataset = DLPretrainDataset(
        roots=roots,
        n_views=args.n_views,
        size=args.image_size,
        filelist_path=args.filelist_path,
        min_scale=args.min_scale,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    # Model
    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
    model = model.to(args.device)

    if torch.cuda.device_count() > 1 and not args.disable_cuda:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    # Optimizer & scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0.0
    )

    simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
    simclr.train(train_loader)


if __name__ == "__main__":
    main()
