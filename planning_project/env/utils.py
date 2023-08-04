# a collection of utility functions specific to environment

import math
import sys, os
import numpy as np
BASE_PATH = os.path.dirname(__file__)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from planning_project.planner.motion_model import motion_model

class SlipDistributionMap:

    def __init__(self, map, smg):
        """
        __init__: 

        :param map: given discretized environment map
        :param smg: slip models generator class
        """
        self.map = map
        self.smg = smg
        self.motion = motion_model()

    def set_slip(self):
        """
        set_slip: set slip observation map 
        
        """
        # init slip map
        slip_map = np.zeros((self.map.n, self.map.n, len(self.motion), 2))
        # for every position
        for y_id_ in range(self.map.n):
            for x_id_ in range(self.map.n):
                slip_map[y_id_, x_id_] = self.get_slip_at_pos(x_id_, y_id_)
        return slip_map

    def get_slip_at_pos(self, x_id: int, y_id: int):
        """
        get_slip_at_pos: get slip observation at given position

        :param x_id: x-axis index of position
        :param y_id: y-axis index of position
        """
        xy_ids = (x_id, y_id)
        grid_id = self.map.calc_grid_id_from_xy_id(xy_ids[0], xy_ids[1])
        # get terrain class
        tf = int(self.map.data.t_class[grid_id])
        # init slip map at pos
        slip_map_ = np.zeros((len(self.motion), 2)) # first col (from the pos), second col (to the pos)
        for i, motion_ in enumerate(self.motion):
            x_id_ = xy_ids[0] + motion_[0]
            y_id_ = xy_ids[1] + motion_[1]
            s_asc = s_des = np.nan
            if 0 <= x_id_ < self.map.n and 0 <= y_id_ < self.map.n:
                # calculate terrain inclination
                pos_n = self.calc_pos_from_xy_id(xy_ids)
                pos_m = self.calc_pos_from_xy_id((x_id_, y_id_))
                dist_xy = np.linalg.norm(pos_n[0:2] - pos_m[0:2])
                dist_z = abs(pos_n[2] - pos_m[2])
                theta = math.degrees(math.atan2(dist_z, dist_xy))
                s_mv_asc = self.smg.observe_noisy_slip(theta, tf)
                s_mv_des = self.smg.observe_noisy_slip(- theta, tf)
                s_asc, s_des = s_mv_asc[0], s_mv_des[0]
            slip_map_[i, 0] = s_asc
            slip_map_[i, 1] = s_des
        return slip_map_

    def calc_pos_from_xy_id(self, xy_ids: tuple):
        """
        calc_pos_from_xy_id: calculate positional information from indices

        :param xy_ids: x- and y-axis indices
        """
        xy_pos = np.array(xy_ids) * self.map.res + np.array([self.map.lower_left_x, self.map.lower_left_y])
        z_pos = self.map.get_value_from_xy_id(xy_ids[0], xy_ids[1])
        pos = np.append(xy_pos, z_pos)
        return pos