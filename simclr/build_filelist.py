# build_filelist.py
from datasets.dl_pretrain_dataset import build_filelist

if __name__ == "__main__":
    roots = [
        "/vast/vs3273/DL_pretrain_500k/train",
        "/vast/vs3273/DL_pretrain_1.5M/pass_96",
        "/vast/vs3273/DL_pretrain_1.5M/openimages_96",
    ]
    filelist_path = "/vast/vs3273/DL_pretrain_1.5M/filelist_simclr_96px.txt"
    build_filelist(roots, filelist_path)
