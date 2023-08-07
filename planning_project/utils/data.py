"""
description: data loader class for network training, validation, and prediction
author: Masafumi Endo
"""

import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data
import os

BASE_PATH = os.path.dirname(__file__)

# helper function for data visualization
def visualize(vmin: float, vmax: float, n_row: int = 1, n_col: int = None, fig: plt.figure = None, **images):
    """
    visualize: visualize input images in one row

    :param vmin: lower bound of visualized pixel infomation
    :param vmax: upper bound of visualized pixel information
    :param n_row: number of rows in fig
    :param fig: matplotlib object
    :param **images: sequence of images (input, mask, prediction results etc...)
    """
    if not fig:
        fig = plt.figure()
        n_row = 1
    if not n_col:
        n_col = len(images)
    for i, (name, image) in enumerate(images.items()):
        ax = fig.add_subplot(n_row, n_col, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(' '.join(name.split('_')))
        if name == 'mask' or 'prediction':
            ax.imshow(image, vmin=vmin, vmax=vmax, cmap="jet")
        if name == 'likelihood':
            ax.imshow(image, vmin=0, vmax=1, cmap="jet")
        else:
            ax.imshow(image)

def create_int_label(data: np.array):
    """
    create_int_label: create integer label from one-hot expression
    
    :param data: one-hot expression label
    """
    label_int = np.argmax(data, axis=2)
    return label_int

class DataSet(data.Dataset):

    def __init__(self, dirname: str, split: str):
        """
        __init__:

        :param dirname: name of directory to map and masked data
        :param split: data split (train, valid, or test)
        """
        self.dirname = dirname
        self.split = split
        self.dirname = os.path.join(self.dirname, self.split + '/')
        self.ids = os.listdir(self.dirname)
        self.ids.remove('seed_info.npy') # remove data that contains seed information 
        self.data_fps = [os.path.join(self.dirname, idx) for idx in self.ids]

    def __len__(self):
        """
        __len__: return number of data

        """
        return len(self.data_fps)

    def __getitem__(self, idx: int):
        """
        __getitem__: get following matrices
        - color_map: colored terrain map as network input
        - mask: masked terrain map as one-hot matrices indicating each terrain feature class

        :param idx: file index
        """
        data = np.load(self.data_fps[idx], allow_pickle=True).item()
        color_map = self.to_tensor(data["input"])
        mask = self.to_tensor(data["label"])
        return color_map, mask

    def get_height_map(self, idx: int):
        """
        __getitem__: get height map for planning paths

        :param idx: file index
        """
        data = np.load(self.data_fps[idx], allow_pickle=True).item()
        height_map = data["height"]
        return height_map

    def to_tensor(self, data: np.array):
        """
        to_tensor: covert input data as (n, n, dim) -> (dim, n, n)

        :param data: input data to be converted
        """
        return data.transpose(2, 0, 1).astype(np.float32)

    def to_image(self, data: np.array):
        """
        to_image: convert input data as (dim, n, n) -> (n, n, dim)

        :param data: input data to be converted
        """
        return data.transpose(1, 2, 0)