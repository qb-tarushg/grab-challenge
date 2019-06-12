import os

import torch
from matplotlib import pyplot as plt
from PIL import Image
from scipy import io as sio
from torch.utils import data as data

# from torchvision import datasets, models, transforms

# import pathlib
# from abc import ABC, abstractmethod
# from typing import Dict


# AUTOTUNE = tf.data.experimental.AUTOTUNE


# def get_image_paths(base_path):
#     data_root = pathlib.Path(base_path)
#     return [str(item) for item in data_root.iterdir()]


# def load_image(path):
#     image = tf.io.read_file(path)
#     return tf.image.decode_jpeg(image, channels=3)


# def preprocess_image(image):
#     image = tf.image.resize(image, [224, 224])
#     image /= 255.0
#     return image


# class Dataset(ABC):
#     @abstractmethod
#     def load_train(self):
#         pass

#     @abstractmethod
#     def load_test(self):
#         pass

#     @abstractmethod
#     def load_train_labels(self):
#         pass

#     @abstractmethod
#     def load_test_labels(self):
#         pass


# class CarDataset(Dataset):
#     def __init__(self, path_dict: Dict):
#         # TODO add path check test
#         self._train_dir_path = path_dict["train_dir"]
#         self._test_dir_path = path_dict["test_dir"]
#         self._train_label_path = path_dict["train_label_dir"]
#         self._test_label_path = path_dict["test_label_dir"]

#     def load_train(self):
#         path_ds = tf.data.Dataset.from_tensor_slices(
#             get_image_paths(self._train_dir_path)
#         )
#         return path_ds.map(load_image, num_parallel_calls=AUTOTUNE)

#     def load_test(self):
#         path_ds = tf.data.Dataset.from_tensor_slices(
#             get_image_paths(self._test_dir_path)
#         )
#         return path_ds.map(load_image, num_parallel_calls=AUTOTUNE)

#     def load_train_labels(self):
#         return tf.data.Dataset.from_tensor_slices(
#             tf.cast(self._get_labels(self._train_label_path), tf.int64)
#         )

#     def load_test_labels(self):
#         return tf.data.Dataset.from_tensor_slices(
#             tf.cast(self._get_labels(self._test_label_path), tf.int64)
#         )

#     def _get_labels(self, path: str) -> np.ndarray:
#         annotations = sio.loadmat(path)["annotations"]
#         _, nlabels = annotations.shape
#         labels = np.array(
#             [int(annotations[:, i][0][4][0]) for i in range(nlabels)]
#         )
#         return labels

#     def get_car_classes(self, path) -> np.ndarray:
#         carsMat = sio.loadmat(path)
#         _, nclasses = carsMat["class_names"].shape
#         cars_classes = np.array(
#             [
#                 (i + 1, carsMat["class_names"][:, i][0][0])
#                 for i in range(nclasses)
#             ]
#         )
#         return cars_classes


def read_image_file(path):
    return torch.from_numpy(plt.imread(path))


class CarDatasetV2(data.Dataset):
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
