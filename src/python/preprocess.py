import os

# import pandas as pd
from matplotlib import pyplot as plt
from scipy import io as sio


def exists(path):
    return os.path.exists(path)


class PreProcess(object):
    """The base class for preprocessing logical steps.
    Args:
        object ([type]): [description]
    """

    def apply(self, **kwargs):
        pass


class FilterNonRGBCarImages(PreProcess):
    """A preprocessing step for Stanford car dataset. This step is represented as to filter non RGB images
    from a pool of images.
    Args:
        img_folder (str): Path of root image folder.
        label_file (str): Path of class label file.
    Raises:
        FileNotFoundError: [description]
    Returns:
        rgb : list of tuples with `(image_name.ext, label)` ext = `jpg, png`
        non_rbg : list of tuples with `(image_name.ext, label)` ext = `jpg, png`
    """

    def __init__(self, img_folder, label_file):
        self.img_folder = os.path.expanduser(img_folder)
        self.label_file = os.path.expanduser(label_file)

        if not (exists(self.img_folder) or exists(self.label_file)):
            raise FileNotFoundError(
                f"Image files doesn't exists {self.img_folder}, {self.label_file}"
            )

        self.rgb = []
        self.non_rgb = []

    def apply(self):
        annotations = sio.loadmat(self.label_file)["annotations"]
        _, nlabels = annotations.shape
        imgfile_labels = (
            (annotations[:, i][0][5][0], int(annotations[:, i][0][4][0]))
            for i in range(nlabels)
        )
        while True:
            record = next(imgfile_labels, None)
            if record is None:
                break
            imgfile, label = record
            img = plt.imread(os.path.join(self.image_folder, imgfile))

            if self._check_rgb(img):
                self.non_rgb.append(record)
            else:
                self.rgb.append(record)
        return self

    # @property
    # def rgb(self):
    #     return self.rgb

    # @property
    # def non_rgb(self):
    #     return self.non_rgb

    def _check_rgb(self, img):
        return len(img.shape) != 3
