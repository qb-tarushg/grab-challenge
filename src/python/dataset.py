import os

import torch
from matplotlib import pyplot as plt
from PIL import Image
from scipy import io as sio
from torch.utils import data as data

# from torchvision import datasets, models, transforms


def read_image_file(path):
    return torch.from_numpy(plt.imread(path))


def split(dataset, split_ratio=0.4):
    total_len = len(dataset)
    first_len = int(split_ratio * total_len)
    second_len = total_len - first_len
    return data.dataset.random_split(dataset, lengths=[first_len, second_len])


class CarDataset(data.Dataset):
    def __init__(
        self,
        file_dir,
        label_dir,
        train=True,
        transform=None,
        label_transform=None,
    ):
        self._img_dir_path = os.path.expanduser(file_dir)
        self._label_dir_path = os.path.expanduser(label_dir)
        self.train = train
        self.transform = transform
        self.label_transform = label_transform
        if not self._check_exists():
            raise FileNotFoundError(
                f"Image files doesn't exists {self._img_dir_path}, {self._label_dir_path}"
            )
        self.training_ds = []
        self.test_ds = []
        annotations = sio.loadmat(self._label_dir_path)["annotations"]
        _, nlabels = annotations.shape
        imgfile_labels = (
            (annotations[:, i][0][5][0], int(annotations[:, i][0][4][0]))
            for i in range(nlabels)
        )

        nonrgb = 0
        files_loaded = 0
        while True:
            record = next(imgfile_labels, None)
            if record is None:
                break
            imgfile, label = record
            data_set = (
                read_image_file(os.path.join(self._img_dir_path, imgfile)),
                label,
            )

            if self._check_rgb(data_set):
                nonrgb += 1
                continue

            if self.train:
                self.training_ds.append(data_set)
            else:
                self.test_ds.append(data_set)
            files_loaded += 1

        print(f"The files with non RGB format = {nonrgb}/{files_loaded}")

    def _check_rgb(self, data_set):
        return len(data_set[0].size()) != 3

    def __getitem__(self, index):
        if self.train:
            img, label = self.training_ds[index]
        else:
            img, label = self.test_ds[index]

        img = Image.fromarray(img.numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return img, label

    def __len__(self):
        return len(self.training_ds) if self.train else len(self.test_ds)

    def _check_exists(self):
        return os.path.exists(self._img_dir_path) or os.path.exists(
            self._label_dir_path
        )


class StanfordCarDataset(data.Dataset):
    """The class represents the stanford car dataset. It assumes the dataset is downloaded.
    Args:
        image_folder (str): The path to image folder where images are kept.
        image_meta (list): List of tuples `(image_name, label)`.
        transform : Transformations which are to be applied on images.
        label_transform : Transformations which are to be applied on labels or class ids.
    """

    def __init__(self, image_folder, image_meta, transform, label_transform):
        self._image_folder = os.path.expanduser(image_folder)
        self._image_meta = image_meta
        self.transform = transform
        self.label_transform = label_transform

        if not self._check_exists():
            raise FileNotFoundError(
                f"Image files doesn't exists {self._image_folder}"
            )

    def _check_exists(self):
        return os.path.exists(self._image_folder)

    def __len__(self):
        return len(self._image_meta)

    def __getitem__(self, index):
        image_name, label = self._image_meta[index]
        img = Image.open(os.path.join(self._image_folder, image_name))

        if self.transform is not None:
            img = self.transform(img)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return img, label
