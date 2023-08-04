"""
description: cost calculation classes including estimator and observer
author: Masafumi Endo
"""

import math
import numpy as np
from scipy.stats import norm

from planning_project.planner.utils import BaseRiskMetrics, GMMRiskMetrics
from planning_project.planner.motion_model import motion_model

class CostEstimator:

    def __init__(self, smg, map, pred, plan_metrics):
        """
        __init__:

        :param smg: slip model generator class
        :param map: map class
        :param pred: predicted terrain class map
        :param plan_metrics: structure containing planning metrics
        """
        self.smg = smg
        self.map = map
        self.pred = pred
        self.type_model = plan_metrics.type_model
        self.type_embed = plan_metrics.type_embed
        self.alpha = plan_metrics.alpha
        # set risk metrics based on type of slip model and uncertainty consideration
        if self.type_model != "gmm":
            self.rm = BaseRiskMetrics(map=self.map,
                                    type_embed=self.type_embed, 
                                    alpha=self.alpha)
        else:
            self.rm = GMMRiskMetrics(smg=self.smg,
                                    map=self.map,
                                    pred=self.pred,
                                    type_embed=self.type_embed,
                                    alpha=self.alpha)

    def calc_cost(self, node_n, node_m, metric: str = "ra-time"):
        """
        calc_cost: calculate cost between two nodes

        :param node_n: nth node
        :param node_m: mth node
        :param metric: cost calculation metric
        """
        dist, theta = self.get_edge_information(node_n, node_m)
        cost_n, is_feasible_n = self.calc_cost_at_node(node_n, dist / 2, theta, metric)
        cost_m, is_feasible_m = self.calc_cost_at_node(node_m, dist / 2, theta, metric)
        if is_feasible_n and is_feasible_m:
            cost = cost_n + cost_m
            return cost, True
        else:
            return None, False

    def calc_cost_at_node(self, node, dist: float, theta: float, metric: str = "ra-time"):
        """
        calc_cost_at_node: calculate half-edge cost

        :param node: node class
        :param dist: half-edge distance
        :param theta: terrain inclination
        :param metric: cost calculation metric
        """
        s, s0 = self.get_slip_at_node(theta, node)
        if metric == "ra-time":
            cost, is_feasible = self.calc_risk_associated_cost(dist, s, s0, theta)
            return cost, is_feasible

    def calc_risk_associated_cost(self, dist: float, s: float, s0: float, theta: float, vel_ref: float = 0.1):
        """
        calc_risk_associated_cost: calculate "risk-associated" time cost

        :param dist: edge distance
        :param s: slip ratio 
        :param s0: slip ratio when theta = 0
        :param theta: terrain inclination
        :param vel_ref: reference velocity
        """
        if -1 < s < 1:
            if 0 <= theta:
                # driving mode
                vel = (1 - s) * vel_ref
            else:
                s = s0 + (s0 - s)
                vel = vel_ref / (s + 1)
            cost = dist / vel
            return cost, True
        else:
            return None, False

    def calc_time(self, dist: float, s: float, theta: float, vel_ref: float = 0.1):
        """
        calc_time: calculate actual time 

        :param dist: edge distance
        :param s: slip ratio 
        :param theta: terrain inclination
        :param vel_ref: reference velocity
        """
        if 0 <= theta:
            # driving mode
            vel = (1 - s) * vel_ref
        else:
            # breaking mode
            vel = vel_ref / (s + 1)
        time = dist / vel
        return time            

    def get_edge_information(self, node_n, node_m):
        """
        get_edge_information: get edge information connecting nth and mth nodes

        :param node_n: nth node
        :param node_m: mth node
        """
        pos_n = self.calc_pos_from_xy_id(node_n.xy_ids)
        pos_m = self.calc_pos_from_xy_id(node_m.xy_ids)
        dist_xy = np.linalg.norm(pos_n[0:2] - pos_m[0:2])
        dist_z = abs(pos_n[2] - pos_m[2])
        dist = np.linalg.norm(pos_n - pos_m)
        theta = math.degrees(math.atan2(dist_z, dist_xy))
        if pos_n[2] <= pos_m[2]:
            # ascent direction
            return dist, theta
        else:
            # descent direction
            return dist, -1 * theta

    def get_slip_at_node(self, theta: float, node):
        """
        get_slip_at_node: get slip ratios at given node

        :param theta: terrain inclination
        :param node: node class
        """
        if self.type_model != "gmm": # single model estimation
            # identify the type of terrain
            grid_id = self.map.calc_grid_id_from_xy_id(node.xy_ids[0], node.xy_ids[1])
            if 0 <= grid_id < self.map.num_grid:
                tf = int(self.pred[grid_id])
            else:
                tf = None
            # get slip at theta in identified terrain type
            if self.type_model == "gtm": # ground truth model
                s_mv = self.smg.get_actual_slip(theta, tf)
                s0_mv = self.smg.get_actual_slip(0, tf)
            elif self.type_model == "obs": # observation
                s_mv = self.smg.observe_noisy_slip(theta, tf)
                s0_mv = self.smg.observe_noisy_slip(0, tf)
            elif self.type_model == "gsm": # gaussian single model
                s_mv = self.smg.predict_slip(theta, tf)
                s0_mv = self.smg.predict_slip(0, tf)
            # embed uncertainty
            s = self.rm.embed_uncertainty(s_mv, theta, node.xy_ids)
            s0 = self.rm.embed_uncertainty(s0_mv, 0, node.xy_ids)
        else: # mixture model estimation
            # embed uncertainty
            s = self.rm.embed_uncertainty(theta, node.xy_ids)
            s0 = self.rm.embed_uncertainty(0, node.xy_ids)
        return s, s0

    def calc_pos_from_xy_id(self, xy_ids: tuple):
        """
        calc_pos_from_xy_id: calculate positional information from indices

        :param xy_ids: x- and y-axis indices
        """
        xy_pos = np.array(xy_ids) * self.map.res + np.array([self.map.lower_left_x, self.map.lower_left_y])
        z_pos = self.map.get_value_from_xy_id(xy_ids[0], xy_ids[1])
        pos = np.append(xy_pos, z_pos)
        return pos

class CostObserver:

    def __init__(self, smg, map):
        """
        __init__:

        :param smg: slip model generator class
        :param map: map class
        """
        self.smg = smg
        self.map = map
        self.motion = motion_model()

    def calc_cost(self, node_n, node_m):
        """
        calc_cost: calculate cost between two nodes

        :param node_n: nth node
        :param node_m: mth node
        """
        dist, theta, motion_ids = self.get_edge_information(node_n, node_m)
        cost_n, is_feasible_n, slip_n = self.calc_cost_at_node(node_n, dist / 2, theta, motion_ids[0])
        cost_m, is_feasible_m, slip_m = self.calc_cost_at_node(node_m, dist / 2, theta, motion_ids[1])
        if is_feasible_n and is_feasible_m:
            cost = cost_n + cost_m
            return cost, True, [slip_n, slip_m]
        else:
            return None, False, [slip_n, slip_m]

    def calc_cost_at_node(self, node, dist: float, theta: float, motion_id: int):
        """
        calc_cost_at_node: calculate half-edge cost

        :param node: node class
        :param dist: half-edge distance
        :param theta: terrain inclination
        :param motion_id: index of motion model
        """
        s_obs = self.get_slip_at_node(node, theta, motion_id)
        if -1 < s_obs < 1:
            cost = self.calc_time(dist, s_obs, theta)
            return cost, True, s_obs
        else:
            if s_obs >= 1:
                s_obs = 1
            elif s_obs <= -1:
                s_obs = -1
            return None, False, s_obs

    def calc_time(self, dist: float, s: float, theta: float, vel_ref: float = 0.1):
        """
        calc_time: calculate actual time 

        :param dist: edge distance
        :param s: slip ratio 
        :param theta: terrain inclination
        :param vel_ref: reference velocity
        """
        if 0 <= theta:
            # driving mode
            vel = (1 - s) * vel_ref
        else:
            # breaking mode
            vel = vel_ref / (s + 1)
        time = dist / vel
        return time            

    def get_edge_information(self, node_n, node_m):
        """
        get_edge_information: get edge information connecting nth and mth nodes

        :param node_n: nth node
        :param node_m: mth node
        """
        motion_x, motion_y = node_m.xy_ids[0] - node_n.xy_ids[0], node_m.xy_ids[1] - node_n.xy_ids[1]
        motion_id_n = self.motion.index([motion_x, motion_y])
        motion_id_m = self.motion.index([- motion_x, - motion_y])
        pos_n = self.calc_pos_from_xy_id(node_n.xy_ids)
        pos_m = self.calc_pos_from_xy_id(node_m.xy_ids)
        dist_xy = np.linalg.norm(pos_n[0:2] - pos_m[0:2])
        dist_z = abs(pos_n[2] - pos_m[2])
        dist = np.linalg.norm(pos_n - pos_m)
        theta = math.degrees(math.atan2(dist_z, dist_xy))
        if pos_n[2] <= pos_m[2]:
            # ascent direction
            return dist, theta, (motion_id_n, motion_id_m)
        else:
            # descent direction
            return dist, -1 * theta, (motion_id_n, motion_id_m)

    def get_slip_at_node(self, node, theta: float, motion_id: int):
        """
        get_slip_at_node: get slip ratios at given node

        :param node: node class
        :param theta: terrain inlination
        :param motion_id: index of motion model
        """
        slip = self.map.data.slip[node.xy_ids[1], node.xy_ids[0]]
        if theta >= 0: # ascent direction
            s_obs = slip[motion_id, 0]
        else: # descent direction
            s_obs = slip[motion_id, 1]
        return s_obs

    def calc_pos_from_xy_id(self, xy_ids: tuple):
        """
        calc_pos_from_xy_id: calculate positional information from indices

        :param xy_ids: x- and y-axis indices
        """
        xy_pos = np.array(xy_ids) * self.map.res + np.array([self.map.lower_left_x, self.map.lower_left_y])
        z_pos = self.map.get_value_from_xy_id(xy_ids[0], xy_ids[1])
        pos = np.append(xy_pos, z_pos)
        return pos