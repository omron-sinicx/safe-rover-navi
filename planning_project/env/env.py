"""
description: a library to generate planetary surface terrain. 
author: Masafumi Endo
"""

import dataclasses
from cmath import sqrt
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns

from scipy import fftpack
import opensimplex as simplex

@dataclasses.dataclass
class Data:
    """
    structure containing map-dependent data

    :param height: terrain height info.
    :param t_class: terrain class info.
    :param color: terrain color info.
    :param slip: slip observation info. (this should be used for path execution)
    """
    height: np.array = None
    t_class: np.array = None
    color: np.array = None
    slip: np.array = None
    name_instance: str = None

class GridMap:

    def __init__(self, n: int, res: float, re: float = 0.8, sigma: float = 10, seed: int = 0):
        """
        __init__:

        :param n: # of grid in one axis
        :param res: grid resolution [m]
        :param re: roughness exponent for fractal surface (0 < re < 1)
        :param sigma: amplitude gain for fractal surface
        :param seed: random seed 
        """
        self.n = n
        self.res = res
        self.c_x = self.n * self.res / 2.0
        self.c_y = self.n * self.res / 2.0
        self.lower_left_x = self.c_x - self.n / 2.0 * self.res
        self.lower_left_y = self.c_y - self.n / 2.0 * self.res

        # generate data array
        self.num_grid = self.n**2
        self.data = Data(height=np.zeros(self.num_grid), t_class=np.zeros(self.num_grid))

        # for fractal surface
        self.re = re
        self.sigma = sigma

        # for heterogeneous terrain features
        self.occ = None

        # fix randomness
        self.seed = seed
        self.set_randomness()

    def set_randomness(self):
        """
        set_randomness: set randomness for reproductivity

        """
        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed)
            simplex.seed(self.seed)
        else:
            self.rng = np.random.default_rng()
            simplex.random_seed()

    # following functions are used for basic operations
    def get_value_from_xy_id(self, x_id: int, y_id: int, field_name: str = "height"):
        """
        get_value_from_xy_id: get values at specified location described as x- and y-axis indices from data structure

        :param x_id: x index
        :param y_id: y index
        :param field_name: name of field in data structure
        """
        grid_id = self.calc_grid_id_from_xy_id(x_id, y_id)

        if 0 <= grid_id < self.num_grid:
            data_ = getattr(self.data, field_name)
            return data_[grid_id]
        else:
            return None

    def get_xy_id_from_xy_pos(self, x_pos: float, y_pos: float):
        """
        get_xy_id_from_xy_pos: get x- and y-axis indices for given positional information

        :param x_pos: x position [m]
        :param y_pos: y position [m]
        """
        x_id = self.calc_xy_id_from_pos(x_pos, self.lower_left_x, self.n)
        y_id = self.calc_xy_id_from_pos(y_pos, self.lower_left_y, self.n)

        return x_id, y_id

    def calc_grid_id_from_xy_id(self, x_id: int, y_id: int):
        """
        calc_grid_id_from_xy_id: calculate one-dimensional grid index from x- and y-axis indices (2D -> 1D transformation)

        :param x_id: x index
        :param y_id: y index
        """
        grid_id = int(y_id * self.n + x_id)
        return grid_id

    def calc_xy_id_from_pos(self, pos: float, lower_left: float, max_id: int):
        """
        calc_xy_id_from_pos: calculate x- or y-axis indices for given positional information

        :param pos: x- or y-axis position
        :param lower_left: lower left information
        :param max_id: max length (width or height)
        """
        id = int(np.floor((pos - lower_left) / self.res))
        assert 0 <= id <= max_id, 'given position is out of the map!'
        return id

    def set_value_from_xy_pos(self, x_pos: float, y_pos: float, val: float, field_name: str = "height"):
        """
        set_value_from_xy_pos: substitute given arbitrary values into data structure at specified x- and y-axis position

        :param x_pos: x position [m]
        :param y_pos: y position [m]
        :param val: arbitrary spatial information
        :param field_name: name of field in data structure
        """
        x_id, y_id = self.get_xy_id_from_xy_pos(x_pos, y_pos)

        if (not x_id) or (not y_id):
            return False

        flag = self.set_value_from_xy_id(x_id, y_id, val, field_name)

        return flag

    def set_value_from_xy_id(self, x_id: int, y_id: int, val: float, field_name: str = "height", is_increment: bool = True):
        """
        set_value_from_xy_id: substitute given arbitrary values into data structure at specified x- and y-axis indices

        :param x_id: x index
        :param y_id: y index
        :param val: arbitrary spatial information
        :param field_name: name of field in data structure
        :param is_increment: increment data if True. Otherwise, simply update value information.
        """
        if (x_id is None) or (y_id is None):
            return False, False

        grid_id = int(y_id * self.n + x_id)

        if 0 <= grid_id < self.num_grid:
            data_ = getattr(self.data, field_name)
            if is_increment:
                data_[grid_id] += val
            else:
                data_[grid_id] = val
            setattr(self.data, field_name, data_)
            return True
        else:
            return False

    # following functions are used for random celestial terrain map genration
    def load_env(self, height_map: np.array, mask: np.array, color_map: np.array):
        """
        load_env: load given terrain information

        :param height_map: terrain height information
        :param mask: terrain class distribution
        :param color_map: corresponding color map
        """
        self.data.height = np.ravel(height_map)
        self.data.t_class = np.ravel(mask)
        self.data.color = color_map

    def set_terrain_env(self, is_crater: bool = True, is_fractal: bool = True, num_crater: int = 5, 
                        min_a: float = 10, max_a: float = 20, min_r: float = 20, max_r: float = 60):
        """
        set_terrain_env: set planetary terrain environment based on fractal method w/ crater

        :param is_crater: existence of crater
        :param is_fractal: existence of terrain roughness
        :param num_crater: number of crater
        :param min_a: min crater slope angle
        :param max_a: max crater slope angle
        :param min_r: min crater range of inner rim
        :param max_r: max crater range of inner rim
        """
        if is_crater:
            i = 0
            while i < num_crater:
                c_xy_ = self.rng.integers(self.lower_left_x, (self.n - 1) * self.res, 2).reshape(1, 2)
                ranges_ = self.rng.integers(min_r, max_r)
                if i == 0:
                    self.set_crater(c_xy=c_xy_, angles=self.rng.integers(min_a, max_a), ranges=ranges_)
                    # init array for checking circle hit
                    c_arr = c_xy_
                    r_arr = np.array([ranges_])
                    i += 1
                else:
                    is_hit = self.check_circle_hit(c_arr, r_arr, c_xy_, ranges_)
                    if not is_hit:
                        self.set_crater(c_xy=c_xy_, angles=self.rng.integers(min_a, max_a), ranges=ranges_)
                        c_arr = np.append(c_arr, c_xy_, axis=0)
                        r_arr = np.append(r_arr, ranges_)
                        i += 1
        if is_fractal:
            self.set_fractal_surf()
        self.data.height = self.set_offset(self.data.height)

    def check_circle_hit(self, c_arr: np.ndarray, r_arr: np.ndarray, c_t: np.ndarray, r_t: np.ndarray):
        """
        check_circle_hit: check whether given craters are overlapped or not

        :param c_arr: array storing circle center
        :param r_arr: array sotring circle radius
        :param c_t: current circle center info.
        :param r_t: current circle radius info.
        """
        for c, r, in zip(c_arr, r_arr):
            dist_c = sqrt((c[0] - c_t[0, 0])**2 + (c[1] - c_t[0, 1])**2)
            if dist_c < max([r, r_t]):
                return True
        return False
        
    def set_fractal_surf(self):
        """
        set_fractal_surf: set fractal surface into data structure

        """
        z = self.generate_fractal_surf()
        # set offset
        z = self.set_offset(np.ravel(z))
        self.data.height += z

    def set_crater(self, c_xy: np.ndarray, angles: np.ndarray, ranges: np.ndarray):
        """
        set_crater: set arbitrary crater generated with given parameters into 2D map environment

        :param c_xy: center of crater position in x- and y-axis [m]
        :param angles: array of inner-rim angles
        :param ranges: array of inner-rim ranges
        """
        if not isinstance(angles, list):
            angles = [angles]
        if not isinstance(ranges, list):
            ranges = [ranges]
        z = self.generate_crater(angles, ranges)
        x_c_id, y_c_id = self.get_xy_id_from_xy_pos(c_xy[0, 0], c_xy[0, 1])
        x_len = int(z.shape[0] / 2)
        y_len = int(z.shape[1] / 2)
        for y_id_ in range(y_c_id - y_len, y_c_id + y_len):
            for x_id_ in range(x_c_id - x_len, x_c_id + x_len):
                if self.lower_left_x <= x_id_ < self.n and self.lower_left_y <= y_id_ < self.n:
                    self.set_value_from_xy_id(x_id_, y_id_, z[x_id_ - (x_c_id - x_len), y_id_ - (y_c_id - y_len)])

    def set_offset(self, z: np.ndarray):
        """
        set_offset: adjust z-axis value starting from zero

        :param z: z-axis information (typically for distance information, such as terrain height)
        """
        z_ = z - min(z)
        z = (z_ + abs(z_)) / 2
        z_ = z - max(z)
        z_ = (z_ - abs(z_)) / 2
        z = z_ - min(z_)
        return z

    def set_terrain_distribution(self, occ: list, type_dist: str = "noise", seed: int = None):
        """
        set_terrain_distribution: set multi class terrain distribution based on given occupancy vector

        :param occ: terrain occupancy
        :param type_dist: type of terrain distribution
        """
        self.occ = occ
        if sum(self.occ) > 1:
            self.occ = (np.array(self.occ) / sum(self.occ)).tolist()
            warnings.warn('sum of occupancy vector exceeds one! the vector is modified...')
        if type_dist == "square":
            data = self.generate_multi_terrain_square(batch_size=None)
        elif type_dist == "noise":
            data = self.generate_multi_terrain_noise(seed=seed)
        self.data.t_class = np.ravel(data)
        self.create_color_map()

    def create_color_map(self):
        """
        create_color_map: create color map based on given terrain class distribution

        """
        facecolors = - np.reshape(self.data.t_class, (self.n, self.n))
        norm = matplotlib.colors.Normalize(vmin=- len(self.occ), vmax=0)
        self.data.color = plt.cm.copper(norm(facecolors))[:, :, 0:3].astype(np.float32)
            
    def generate_fractal_surf(self):
        """
        generate_fractal_surf: generate random height information based on fractional Brownian motion (fBm).

        """
        z = np.zeros((self.n, self.n), complex)
        for y_id_ in range(int(self.n / 2) + 1):
            for x_id_ in range(int(self.n / 2) + 1):
                phase = 2 * np.pi * self.rng.random()
                if x_id_ != 0 or y_id_ != 0:
                    rad = 1 / (x_id_**2 + y_id_**2)**((self.re + 1) / 2)
                else:
                    rad = 0.0
                z[y_id_, x_id_] = rad * np.exp(1j * phase)
                if x_id_ == 0:
                    x_id_0 = 0
                else:
                    x_id_0 = self.n - x_id_
                if y_id_ == 0:
                    y_id_0 = 0
                else:
                    y_id_0 = self.n - y_id_
                z[y_id_0, x_id_0] = np.conj(z[y_id_, x_id_])

        z[int(self.n / 2), 0] = np.real(z[int(self.n / 2), 0])
        z[0, int(self.n / 2)] = np.real(z[0, int(self.n / 2)])
        z[int(self.n / 2), int(self.n / 2)] = np.real(z[int(self.n / 2), int(self.n / 2)])

        for y_id_ in range(1, int(self.n / 2)):
            for x_id_ in range(1, int(self.n / 2)):
                phase = 2 * np.pi * self.rng.random()
                rad = 1 / (x_id_ ** 2 + y_id_ ** 2) ** ((self.re + 1) / 2)
                z[y_id_, self.n - x_id_] = rad * np.exp(1j * phase)
                z[self.n - y_id_, x_id_] = np.conj(z[y_id_, self.n - x_id_])

        z = z * abs(self.sigma) * (self.n * self.res * 1e+3)**(self.re + 1 + .5)
        z = np.real(fftpack.ifft2(z)) / (self.res * 1e+3)**2
        z = z * 1e-3
        return z

    def generate_crater(self, angles: np.ndarray, ranges: np.ndarray):
        """
        generate_crater: generate crater height information

        :param angles: array of inner-rim angles
        :param ranges: array of inner-rim ranges
        """
        xx, yy = np.meshgrid(np.arange(0.0, 2.0 * sum(ranges), self.res),
                             np.arange(0.0, 2.0 * sum(ranges), self.res))
        c_x = c_y = sum(ranges)
        r_btm = 0
        dh = 0
        for i, (a, r) in enumerate(zip(angles, ranges)):
            r_top = r_btm + r
            rr = np.sqrt((xx - c_x)**2 + (yy - c_y)**2)
            h = r_top * np.tan(np.radians(a))
            z = (h / r_top) * (rr - r_top)
            z = z - np.min(z)
            dh_btm = r_btm * np.tan(np.radians(a))
            dh_top = r_top * np.tan(np.radians(a))
            if i != 0:
                z += dh - dh_btm
                z[rr < r_btm] = z_p[rr < r_btm]
                z[rr >= r_top] = np.nan
            else:
                z[rr >= r_top] = np.nan
            z_p = z
            dh += dh_top - dh_btm
        z[rr >= r_top] = h -dh_btm
        z -= abs(np.max(z))

        return z

    def generate_multi_terrain_square(self, batch_size: int = None):
        """
        generate_multi_terrain_square: generate terrain class data based on the given occupancy information

        :param batch_size: size of minibatch
        """
        if batch_size is None:
            batch_size = int(self.n / 5)
        n_square = int(self.num_grid / batch_size**2)
        occ = np.array(self.occ) * n_square
        squares = []
        for i, num in enumerate(occ):
            square = np.ones((batch_size, batch_size)) * i
            for _ in range(int(num)):
                squares.append(square)
        n_batch = int(self.n / batch_size)
        data = None
        self.rng.shuffle(squares)
        for i in range(n_batch):
            data_h = np.hstack(squares[(n_batch * i):(n_batch * (i + 1))])
            if data is None:
                data = data_h
            else:
                data = np.vstack((data, data_h))
        return data

    def generate_multi_terrain_noise(self, f_size: float = 20, res_step: float = 1, seed: int = None):
        """
        generate_multi_terrain_noise: generate terrain class data based on noise given by open-simplex

        :param f_size: feature size
        :param res_step: increment step resolution (smaller value -> highly accurate occupancy ratio)
        :param seed: seed to be set for random shufulle function
        """
        noise_data = np.zeros((self.n, self.n))
        for y_id_ in range(self.n):
            for x_id_ in range(self.n):
                noise_data[y_id_][x_id_] = simplex.noise2(x_id_ / f_size, y_id_ / f_size)
        noise_data -= np.min(noise_data)
        noise_data = noise_data / np.max(noise_data) * 100

        occ = np.array(self.occ) * self.num_grid
        step_ub = 0
        step_lb = -1 # lower bound should be lower than min(noise_data)
        tf = list(range(occ.size))
        if seed is not None:
            random.seed(seed)
            random.shuffle(tf)
            occ = [occ[i] for i in tf]
        terrain_data = np.full((self.n, self.n), np.nan)
        i_tf = 0 # index of terrain features
        while step_ub <= np.max(noise_data):
            if occ[i_tf] == 0:
                # skip terrain feature assignment if zero occupancy ratio
                i_tf += 1
                continue
            n_grid_ = sum(sum(np.where(noise_data <= step_ub, 1, 0)))
            if n_grid_ >= sum(occ[:i_tf + 1]) - 1e-8: # avoid floating error
                if i_tf == len(occ)-1:
                    terrain_data[step_lb < noise_data] = tf[i_tf]
                else:
                    terrain_data[(step_lb < noise_data) & (noise_data <= step_ub)] = tf[i_tf]
                step_lb = step_ub
                i_tf += 1
            step_ub += res_step
        return terrain_data

    def extend_data(self, data: np.array, field_name):
        """
        extend_data: extend self.data in case additional terrain features are necessary

        :param data: appended data array
        """
        setattr(self.data, field_name, data)

    # following functions are used for visualization objective
    def print_grid_map_info(self):
        """
        print_grid_map_info: show grid map information

        """
        print("range: ", self.n * self.res, " [m]")
        print("resolution: ", self.res, " [m]")
        print("# of data: ", self.num_grid)

    def plot_maps(self, figsize: tuple = (10, 4), is_tf: bool = True):
        """
        plot_maps: plot 2D and 2.5D figures with given size

        :param figsize: size of figure
        :param is_tf: existence of terrain features
        """
        sns.set()
        sns.set_style('whitegrid')
        fig = plt.figure(figsize=figsize)
        _, ax_3d = self.plot_3d_map(fig=fig, rc=121, is_tf=is_tf)
        _, ax_2d = self.plot_2d_map(fig=fig, rc=122)
        plt.tight_layout()
        return ax_2d, ax_3d

    def plot_2d_map(self, grid_data: np.ndarray = None, fig: plt.figure = None, rc: int = 111, field_name: str = "height", 
                    title: str = "2D celestial terrain", cmap: str = "jet", label: str = "height m"):
        """
        plot_2d_map: plot 2D grid map

        :param grid_data: data to visualize
        :param fig: figure
        :param rc: position specification as rows and columns
        :param i_tf: index of terrain features
        :param title: title of shown figure
        """
        xx, yy = np.meshgrid(np.arange(0.0, self.n * self.res, self.res),
                             np.arange(0.0, self.n * self.res, self.res))

        if grid_data is None:
            grid_data = np.reshape(getattr(self.data, field_name), (self.n, self.n))
            data = getattr(self.data, field_name)
        else:
            data = np.reshape(grid_data, -1)
        if not fig:
            fig = plt.figure()

        ax = fig.add_subplot(rc)
        hmap = ax.pcolormesh(xx + self.res / 2.0, yy + self.res / 2.0, grid_data,
                             cmap=cmap, vmin=min(data), vmax=max(data))
        ax.set_xlabel("x-axis m")
        ax.set_ylabel("y-axis m")
        ax.set_aspect("equal")
        ax.set_xlim(xx.min(), xx.max() + self.res)
        ax.set_ylim(yy.min(), yy.max() + self.res)
        ax.set_title(title)

        plt.colorbar(hmap, ax=ax, label=label, orientation='horizontal')

        return hmap, ax

    def plot_3d_map(self, grid_data: np.ndarray = None, fig: plt.figure = None, rc: int = 111, is_tf: bool = False):
        """
        plot_3d_map: plot 2.5D grid map

        :param grid_data: data to visualize
        :param fig: figure
        :param rc: position specification as rows and columns
        :param is_tf: existence of terrain features
        """
        xx, yy = np.meshgrid(np.arange(0.0, self.n * self.res, self.res),
                             np.arange(0.0, self.n * self.res, self.res))

        if grid_data is None:
            grid_data = np.reshape(self.data.height, (self.n, self.n))
            data = self.data.height
        else:
            data = np.reshape(grid_data, -1)
        if not fig:
            fig = plt.figure()

        ax = fig.add_subplot(rc, projection="3d")
        if not is_tf:
            hmap = ax.plot_surface(xx + self.res / 2.0, yy + self.res / 2.0, grid_data,
                                cmap="jet", vmin=min(data), vmax=max(data), linewidth=0, antialiased=False)
        else:
            hmap = ax.plot_surface(xx + self.res / 2.0, yy + self.res / 2.0, grid_data, 
                                facecolors=self.data.color, linewidth=0, antialiased=False)
        ax.set_xlabel("x-axis m")
        ax.set_ylabel("y-axis m")
        ax.set_zticks(np.arange(xx.min(), xx.max(), 10))
        ax.view_init(elev=30, azim=45)
        ax.set_box_aspect((1, 1, 0.25))
        ax.set_xlim(xx.min(), xx.max() + self.res)
        ax.set_ylim(yy.min(), yy.max() + self.res)
        ax.set_zlim(min(data), xx.max() / 10)
        ax.set_title("2.5D celestial terrain")

        return hmap, ax