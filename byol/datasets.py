# datasets.py

import os
from glob import glob
from typing import Callable, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# class BYOLPretrainDataset(Dataset):
#     """
#     Unlabeled pretraining dataset.
#     Expects one or more root directories with images (optionally nested).
#     """

#     def __init__(self, root_dirs, transform1: Callable, transform2: Callable):
#         super().__init__()

#         if isinstance(root_dirs, str):
#             root_dirs = [root_dirs]
#         self.root_dirs = root_dirs

#         exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.webp")
#         files: List[str] = []

#         for root_dir in root_dirs:
#             root_files = []
#             for ext in exts:
#                 pattern = os.path.join(root_dir, "**", ext)
#                 root_files.extend(glob(pattern, recursive=True))
#             print(f"[BYOLPretrainDataset] Root {root_dir} has {len(root_files)} images")
#             files.extend(root_files)

#         if len(files) == 0:
#             raise RuntimeError(f"No images found in: {root_dirs}")

#         self.files = sorted(files)
#         print(
#             f"[BYOLPretrainDataset] Found {len(self.files)} images "
#             f"from {len(root_dirs)} roots."
#         )

#         self.transform1 = transform1
#         self.transform2 = transform2

#     def __len__(self) -> int:
#         return len(self.files)

#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         path = self.files[idx]
#         img = Image.open(path).convert("RGB")
#         v1 = self.transform1(img)
#         v2 = self.transform2(img)
#         return v1, v2

class BYOLPretrainDataset(Dataset):
    """
    Unlabeled pretraining dataset.
    Expects one or more root directories with images (optionally nested).

    Uses a cached file list if provided, so startup is fast on GPU jobs.
    """

    def __init__(
        self,
        root_dirs,
        transform1: Callable,
        transform2: Callable,
        cache_list_path: str = None,
    ):
        super().__init__()

        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
        self.root_dirs = root_dirs
        self.transform1 = transform1
        self.transform2 = transform2

        exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")

        # If cache exists, use it
        if cache_list_path is not None and os.path.exists(cache_list_path):
            print(f"[BYOLPretrainDataset] Loading file list from cache: {cache_list_path}")
            with open(cache_list_path, "r") as f:
                files = [line.strip() for line in f if line.strip()]
        else:
            # Fallback: build file list (should not happen often on GPU jobs)
            files: List[str] = []
            for root_dir in root_dirs:
                root_files = []
                for dirpath, _, filenames in os.walk(root_dir):
                    for fname in filenames:
                        if fname.lower().endswith(exts):
                            full = os.path.join(dirpath, fname)
                            root_files.append(full)
                print(f"[BYOLPretrainDataset] Root {root_dir} has {len(root_files)} images")
                files.extend(root_files)

            # Optional: save cache if path provided
            if cache_list_path is not None:
                os.makedirs(os.path.dirname(cache_list_path), exist_ok=True)
                with open(cache_list_path, "w") as f:
                    for p in files:
                        f.write(p + "\n")
                print(f"[BYOLPretrainDataset] Saved file list to {cache_list_path}")

        if len(files) == 0:
            raise RuntimeError(f"No images found in: {root_dirs}")

        self.files = sorted(files)
        print(
            f"[BYOLPretrainDataset] Found {len(self.files)} images "
            f"from {len(root_dirs)} roots."
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        v1 = self.transform1(img)
        v2 = self.transform2(img)
        return v1, v2




def get_byol_transforms(image_size: int = 96):
    """
    BYOL / SimCLR-style augmentations adapted to 96x96 images.

    Two pipelines:
      - view1: strong blur (p=1.0), no solarization
      - view2: weaker blur (p=0.1), solarization (p=0.2)
    """

    # Color jitter parameters (brightness, contrast, saturation, hue)
    color_jitter = T.ColorJitter(0.4, 0.4, 0.2, 0.1)

    base_transforms = [
        T.RandomResizedCrop(
            image_size,
            scale=(0.4, 1.0),
            ratio=(0.75, 1.33),
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([color_jitter], p=0.8),
        T.RandomGrayscale(p=0.2),
    ]

    # Gaussian blur kernel size roughly 0.1 * image_size
    kernel_size = int(0.1 * image_size)
    if kernel_size % 2 == 0:
        kernel_size += 1  # must be odd

    blur = T.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0))

    # Solarization (torchvision has it in transforms)
    solarize = T.RandomSolarize(threshold=0.5, p=1.0)

    transform_view1 = T.Compose(
        base_transforms
        + [
            # always blur, no solarization
            T.RandomApply([blur], p=1.0),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    transform_view2 = T.Compose(
        base_transforms
        + [
            # sometimes blur, sometimes solarize
            T.RandomApply([blur], p=0.1),
            T.RandomApply([solarize], p=0.2),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return transform_view1, transform_view2
