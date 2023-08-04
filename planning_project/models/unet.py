"""
description: U-net model for multi-class terrain classification based on image semantic segmentation 
author: Masafumi Endo
"""

import segmentation_models_pytorch as smp

class Unet:

    def __init__(self, n_terrains: int, in_channels: int):
        """
        __init__:

        :param n_terrains: number of terrain classes
        :param in_channels: model input channels (3 for RGB, 1 for gray-scale, etc.)
        """
        self.n_terrains = n_terrains
        self.in_channels = in_channels

    def set_model(self):
        return smp.Unet(encoder_name="resnet18",
                        encoder_weights="imagenet",
                        classes=self.n_terrains,
                        in_channels=self.in_channels,
                        activation="softmax2d",
                        )