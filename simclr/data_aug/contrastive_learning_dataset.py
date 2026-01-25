# data_aug/contrastive_learning_dataset.py
from torchvision import transforms, datasets
from data_aug.gaussian_blur import GaussianBlur
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1, min_scale=0.2):
        """
        Return a set of data augmentation transformations as described in
        the SimCLR paper, but allow controlling the minimum crop scale.

        min_scale: minimum proportion of image area in the crop.
        For 96x96 images, 0.2 means ~43x43 minimum crop (not too tiny).
        """
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(
                size=size,
                scale=(min_scale, 1.0),
                ratio=(3 / 4, 4 / 3),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=int(0.1 * size)),
            transforms.ToTensor(),
        ])
        return data_transforms

    def get_dataset(self, name, n_views):
        """
        Existing CIFAR10/STL10 options kept for reference.
        We won't actually use them for your pretrain, but they still work.
        """
        valid_datasets = {
            'cifar10': lambda: datasets.CIFAR10(
                self.root_folder, train=True,
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(size=32),
                    n_views
                ),
                download=True
            ),
            'stl10': lambda: datasets.STL10(
                self.root_folder, split='unlabeled',
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(size=96),
                    n_views
                ),
                download=True
            ),
        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()

