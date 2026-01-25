from datasets import BYOLPretrainDataset, get_byol_transforms

root_dirs = [
    "/scratch/vs3273/DL_pretrain_500k/train",
    "/scratch/vs3273/DL_pretrain_1.5M/pass_96k",
    "/scratch/vs3273/DL_pretrain_1.5M/openimages_96k",
]

t1, t2 = get_byol_transforms(image_size=96)
ds = BYOLPretrainDataset(root_dirs, t1, t2)
print("Total images:", len(ds))
v1, v2 = ds[0]
print("View1 shape:", v1.shape, "View2 shape:", v2.shape)