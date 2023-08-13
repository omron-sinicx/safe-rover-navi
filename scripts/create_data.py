"""
description: data preparation scripts 
author: Masafumi Endo
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import sys, os

BASE_PATH = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_PATH, '../planning_project'))

from planning_project.env.env import GridMap
from planning_project.env.slip_models import SlipModel, SlipModelsGenerator
from planning_project.utils.viz import viz_terrain_props

class DataGenerator:

    def __init__(self, n_ds: str, split: str, n: int, n_envs: int = 10, n_terrains: int = 10, n_colors: int = 5, n_instances: int = 100, is_show: bool = False, is_random: bool = False):
        """
        __init__: 

        :param n_data: name of datasets, as [Std, ES, AA]
        :param split: data split (train, valid, or test)
        :param n: # of grid in one axis
        :param n_envs: number of environment types that have different occupancy vector
        :param n_terrains: number of terrain features
        :param n_colors: number of terrain colors
        :param n_instances: number of instances of each environment types
        :param is_show: show generated terrain map or not
        :param is_random: set randomness (True: no reproductivity, False: reproductivity)
        """
        self.n_ds = n_ds
        self.split = split
        self.n = n
        self.n_terrains = n_terrains
        self.n_colors = n_colors
        self.n_envs = n_envs
        self.n_instances = n_instances
        self.is_show = is_show
        self.is_random = is_random

        self.dict_ds = {"Std": 1, "ES": 2, "AA": 3}
        # make directory if not
        path2ds = os.path.join(BASE_PATH, '../datasets/dataset_%s/%s/' % (self.n_ds, self.split))
        if not os.path.exists(path2ds):
            os.makedirs(path2ds, exist_ok=True)
        # call smg
        if n_ds == "Std":
            type_noise = 'uniform'
        else:
            type_noise = 'diverse'
        self.smg = SlipModelsGenerator(os.path.join(BASE_PATH, '../datasets/dataset_%s/' % (self.n_ds)), self.n_terrains, is_update=True, type_noise=type_noise)

        # set conditions for occupancy ratio
        self.n_filled = 4
        self.filled_ratios = [0.4, 0.1] # major
        if self.n_terrains != self.n_colors:
            self.n_filled = None
            self.filled_ratios = None
            self.terrain_pairs = [] # store pairs of terrain classes (either low and high slip terrain)
            self.get_terrain_pairs()

    def create_data(self):
        """
        create_data: create datasets for training ML architecture

        """
        # initialize seed for environment types and instances
        seed_envs, seed_instances = self.set_seed_info()
        for i in range(self.n_envs):
            if self.is_random:
                seed_envs = None
            else:
                seed_envs += 1
            if self.n_terrains == self.n_colors:
                occ = self.set_terrain_occ(seed=seed_envs, n_terrains=self.n_terrains)
            else:
                occ = [1 / self.n_colors] * self.n_colors
            print("generate map with occupancy vector: ", occ, ", seed: ", seed_envs)
            for j in range(self.n_instances):
                if self.is_random:
                    seed_instances = None
                else:
                    seed_instances += 1
                grid_map = GridMap(self.n, 1, seed=seed_instances)
                grid_map.set_terrain_env(is_crater=True, is_fractal=True, num_crater=5, max_a=20, max_r=25)
                height_map = np.reshape(grid_map.data.height, (grid_map.n, grid_map.n))
                # set terrain distribution (decide color distribution)
                if self.n_terrains != self.n_colors:
                    grid_map.set_terrain_distribution(occ=occ, type_dist="noise", seed=seed_envs)
                    # generate noise information that control distribution of two terrains
                    t_class = self.generate_complex_color_map(grid_map, seed_instances)
                    grid_map.data.t_class = t_class
                else:
                    grid_map.set_terrain_distribution(occ=occ, type_dist="noise")
                color_map = grid_map.data.color
                label_one_hot = np.reshape(self.create_one_hot_label(grid_map.data.t_class), (grid_map.n, grid_map.n, self.n_terrains))
                print("map generation done for #", j+1, " instances, seed: ", seed_instances)
                if self.is_show:
                    viz_terrain_props(vmin=0, vmax=9, n_row=1, n_col=2, fig=plt.figure(),
                            terrain_color=grid_map.data.color,
                            mask=np.reshape(grid_map.data.t_class, (grid_map.n, grid_map.n)))
                    plt.show()
                np.save(os.path.join(BASE_PATH, '../datasets/dataset_%s/%s/env_%02d_%04d' % (self.n_ds, self.split, i, j)), {'input': color_map, 'label': label_one_hot, 'height': height_map})
        seed_info = {'seed_envs': seed_envs, 'seed_instances': seed_instances}
        np.save(os.path.join(BASE_PATH, '../datasets/dataset_%s/%s/seed_info' % (self.n_ds, self.split)), seed_info)

    def set_seed_info(self):
        """
        set_seed_info: set counter of seeds considering reproductivity

        """
        if self.split == "train":
            if self.n_ds == "Std" or self.n_ds == "ES":
                seed_envs = 0
                seed_instances = 0
            else:
                if self.n_ds == "AA":
                    n_ds_p = "ES"
                seed_info = np.load(os.path.join(BASE_PATH, '../datasets/dataset_%s/test/seed_info.npy' % (n_ds_p)), allow_pickle=True).item()
                seed_envs = seed_info["seed_envs"]
                seed_instances = seed_info["seed_instances"]
        elif self.split == "valid":
            seed_info = np.load(os.path.join(BASE_PATH, '../datasets/dataset_%s/train/seed_info.npy' % (self.n_ds)), allow_pickle=True).item()
            seed_envs = seed_info["seed_envs"]
            seed_instances = seed_info["seed_instances"]
        elif self.split == "test":
            seed_info = np.load(os.path.join(BASE_PATH, '../datasets/dataset_%s/valid/seed_info.npy' % (self.n_ds)), allow_pickle=True).item()
            seed_envs = seed_info["seed_envs"]
            seed_instances = seed_info["seed_instances"]
        print("set seed information! for environments: ", seed_envs, ", for instances: ", seed_instances)
        return seed_envs, seed_instances

    def get_terrain_pairs(self):
        """
        get_terrain_pairs: get pairs of terrains sharing same colors

        """
        # get latent slip model gain
        random.seed(self.dict_ds[self.n_ds])
        gains = [slip_model.gain for slip_model in self.smg.slip_models]
        gains_sorted = sorted(gains)
        gains_l = random.sample(gains_sorted[:self.n_colors], self.n_colors)
        gains_h = random.sample(gains_sorted[self.n_colors:], self.n_colors)
        for i in range(self.n_colors):
            class_l = gains.index(gains_l[i])
            class_h = gains.index(gains_h[i])
            terrain_pair = [class_l, class_h]
            self.terrain_pairs.append(terrain_pair)

    def generate_complex_color_map(self, grid_map, seed):
        """
        generate_complex_color_map: generate color map that contains different terrain class but same color
        
        :param grid_map: map information
        :param seed: seed of instances
        """
        occ_ = [0.5] * int(self.n_terrains / self.n_colors)
        grid_map_ = GridMap(self.n, 1, seed=seed + 100000)
        grid_map_.occ = occ_
        data = np.ravel(grid_map_.generate_multi_terrain_noise(f_size=4))
        # pair [0, 1] = [low, high] slip terrain
        t_class = np.copy(grid_map.data.t_class)
        t_class_new = np.full(t_class.size, np.inf)
        for i, terrain_pair in enumerate(self.terrain_pairs):
            t_class_new[(t_class == i) & (data == 0)] = terrain_pair[0]
            t_class_new[(t_class == i) & (data == 1)] = terrain_pair[1]
        return t_class_new

    def set_terrain_occ(self, seed: int, n_terrains: int):
        """
        set_terrain_occ: set terrain occupancy vector 

        :param seed: random seed
        """
        random.seed(seed)
        tf = list(range(n_terrains))
        tf_filled = random.sample(tf, self.n_filled)
        tf_filled_major = random.sample(tf_filled, 2) # major terrain (occupy 40 % in map)
        occ = [0] * n_terrains
        for i in range(n_terrains):
            if i in tf_filled:
                if i in tf_filled_major:
                    occ[i] = self.filled_ratios[0] # major tarrain
                else:
                    occ[i] = self.filled_ratios[1] # minor terrain
        return occ

    def create_one_hot_label(self, data: np.array):
        """
        create_one_hot_labels: create one-hot labels based on integer terrain label expression

        :param data: terrain feature data
        """
        label_one_hot = np.eye(self.n_terrains)[data.astype(int)]
        return label_one_hot

def create_datasets(n_ds: str):
    # set terrain classes and corresponding color info.
    if n_ds == "Std" or n_ds == "ES":
        n_terrains = 10
        n_colors = 10
    elif n_ds == "AA":
        n_terrains = 8
        n_colors = 4
    data_generator_train = DataGenerator(n_ds=n_ds, split="train", n=96, n_envs=10, n_terrains=n_terrains, n_colors=n_colors, n_instances=100, is_show=False, is_random=False)
    data_generator_valid = DataGenerator(n_ds=n_ds, split="valid", n=96, n_envs=10, n_terrains=n_terrains, n_colors=n_colors, n_instances=50, is_show=False, is_random=False)
    data_generator_test = DataGenerator(n_ds=n_ds, split="test", n=96, n_envs=10, n_terrains=n_terrains, n_colors=n_colors, n_instances=10, is_show=False, is_random=False)
    data_generator_train.create_data()
    data_generator_valid.create_data()
    data_generator_test.create_data()

def main():
    n_ds = "Std"
    create_datasets(n_ds=n_ds)

if __name__ == '__main__':
    main()