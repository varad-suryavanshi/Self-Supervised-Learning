# build_filelist_1p05M.py
import os

ROOTS = [
    "/scratch/vs3273/DL_pretrain_500k/train",
    "/scratch/vs3273/DL_pretrain_1.5M/pass_96",
    "/scratch/vs3273/DL_pretrain_1.5M/openimages_96",
]

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")
OUT_PATH = "/scratch/vs3273/DL_pretrain_1.5M/filelist_1p05M.txt"

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

all_files = []

for root in ROOTS:
    root = os.path.abspath(root)
    count = 0
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(EXTS):
                full = os.path.join(dirpath, fname)
                all_files.append(full)
                count += 1
    print(f"[build_filelist] Root {root} has {count} images")

print(f"[build_filelist] Total images: {len(all_files)}")

with open(OUT_PATH, "w") as f:
    for p in all_files:
        f.write(p + "\n")

print(f"[build_filelist] Saved file list to {OUT_PATH}")
