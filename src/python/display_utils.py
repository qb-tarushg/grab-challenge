import os
from os.path import join

import numpy as np
from matplotlib import pyplot as plt


def display(folder, nrows=5, ncols=2, figsize=(50, 50), verbose=False):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for i, ax in enumerate(axes.flat):
        ax.imshow(plt.imread(_get_random_image_path(folder)))


def _get_random_image_path(base_path):
    files_dir = os.listdir(base_path)
    idx = np.random.randint(1, len(files_dir))
    return join(base_path, files_dir[idx])
