"""
description: a planner class based on the conventional A* algorithm. 
author: Masafumi Endo
"""

import heapq
import sys, os
BASE_PATH = os.path.dirname(__file__)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from env.env import Data, GridMap
from env.slip_models import SlipModel, SlipModelsGenerator
from planning_project.planner.utils import Metrics
from planning_project.planner.cost_calculator import CostEstimator, CostObserver
from planning_project.planner.motion_model import motion_model
from planning_project.utils.data import DataSet, visualize, create_int_label

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Node:

    def __init__(self, xy_ids: tuple, node_p):
        """
        __init__:

        :param xy_ids: tuple of x- and y-axis indices
        :param node_p: parent node
        """
        self.xy_ids = xy_ids
        self.node_p = node_p
        
        self.f = 0
        self.g = 0
        self.h = 0

    def __eq__(self, other):
        """
        __eq__: check equality of two nodes' positions

        """
        return self.xy_ids == other.xy_ids

    def __lt__(self, other):
        """
        __lt__: compare values
        
        """
        return self.f < other.f

    def __repr__(self):
        """
        __repr__: show cost for pq operation

        """
        return f'cost at node: {self.f}'

class PriorityQueue:

    def __init__(self):
        """
        __init__: 

        """
        self.nodes = []

    def insert(self, node):
        """
        insert: append node into open/closed list

        :param node: appended node
        """
        self.nodes.append(node)

    def update(self, node):
        """
        update: update cost value of the already existing node
        
        """
        node_id = self.nodes.index(node)
        if node.f < self.nodes[node_id].f:
            self.nodes[node_id] = node

    def pop(self):
        """
        pop: find best node and remove it from open/closed list

        """
        node_best = self.nodes[0]
        node_id = 0
        for i, node_ in enumerate(self.nodes):
            if node_.f < node_best.f:
                node_best = node_
                node_id = i
        self.nodes.pop(node_id)
        return node_best

    def test(self, node):
        """
        test: check the node is already appended open/closed list, or not

        :param node: tested node
        """
        if node in self.nodes:
            return True
        else:
            return False

class PriorityHeapQueue:
    
    def __init__(self):
        """
        __init__: 

        """
        self.nodes = []

    def insert(self, node):
        """
        insert: append node into open/closed list

        :param node: appended node
        """
        heapq.heappush(self.nodes, node)

    def update(self, node):
        """
        update: update cost value of the already existing node
        
        """
        node_id = self.nodes.index(node)
        if node.f < self.nodes[node_id].f:
            self.nodes[node_id] = node
            heapq.heapify(self.nodes)

    def pop(self):
        """
        pop: find best node and remove it from open/closed list

        """
        node_best = heapq.heappop(self.nodes)
        return node_best

    def test(self, node):
        """
        test: check the node is already appended open/closed list, or not

        :param node: tested node
        """
        if node in self.nodes:
            return True
        else:
            return False

class AStarPlanner:

    def __init__(self, map, smg, hyper_params):
        """
        __init__:

        :param map: given discretized environment map
        :param smg: slip models generator class
        :param hyper_params: hyper parameter structure
        """
        self.hyper_params = hyper_params
        self.smg = smg
        self.nn_model_dir = self.hyper_params.nn_model_dir

        if map is None:
            # init w/o map information
            self.map = None
            self.cost_estimator = None
            self.cost_observer = None
        else:
            # init w/ map information
            self.map = map
            self.cost_estimator = None # estimator for path planning
            self.cost_observer = CostObserver(self.smg, self.map)

        self.motion = motion_model()

        self.node_start = None
        self.node_goal = None
        self.node_failed = None
        self.path = None
        self.nodes = []

        self.pred = None
        self.pred_prob = None

    def reset(self, map, start_pos: tuple, goal_pos: tuple, plan_metrics):
        """
        reset: reset position and the use of NN prediction

        :param map: new problem instance
        :param start_pos: start position [m]
        :param goal_pos: goal position [m]
        :param plan_metrics: structure containing planning metrics
        """
        # prepare map and prediction
        self.map = map
        if plan_metrics.type_model != "gtm":
            _ = self.predict(self.map.data.color.transpose(2, 0, 1).astype(np.float32))

        self.cost_estimator = self.set_cost_estimator(plan_metrics) # estimator for path planning
        self.cost_observer = CostObserver(self.smg, self.map)

        self.node_start = Node((self.map.get_xy_id_from_xy_pos(start_pos[0], start_pos[1])), None)
        self.node_goal = Node((self.map.get_xy_id_from_xy_pos(goal_pos[0], goal_pos[1])), None)
        self.node_failed = None
        self.path = None
        self.nodes = []

        self.pred = None
        self.pred_prob = None

    def predict(self, color_map):
        """
        predict: predict terrain types via trained networks
        
        :param color_map: given color map for network prediction
        """
        best_model = torch.load(self.nn_model_dir, map_location=torch.device(DEVICE))
        input = torch.tensor(color_map).unsqueeze(0)
        pred = best_model(input.to(DEVICE)) # torch.size([1, dim, n, n])
        self.pred_prob = pred[0].cpu().detach().numpy() # np.array([dim, n, n]) -> probability prediction
        pred = np.argmax(self.pred_prob, axis=0) # np.array([n, n]) -> distinctive prediction
        self.pred = np.ravel(pred)
        return self.pred

    def set_cost_estimator(self, plan_metrics):
        """
        set_cost_estimator: set cost estimator class 

        :param plan_metrics: structure containing planning metrics
        """
        if plan_metrics.type_model == "gtm": # ground truth model
            pred = self.map.data.t_class
        elif plan_metrics.type_model == "gsm": # gaussian single model
            pred = self.pred
        elif plan_metrics.type_model == "gmm": # gaussian mixture model
            pred = self.pred_prob
        cost_estimator = CostEstimator(self.smg, self.map, pred, plan_metrics)
        return cost_estimator
    
    def search_path(self):
        """
        search_path: main loop of A* search algorithm

        :param start_pos: start position [m]
        :param goal_pos: goal position [m]
        :param type_model: type of model estimation
        :param type_embed: type to embed uncertainty int cost
        """
        open_list = PriorityHeapQueue()
        closed_list = PriorityHeapQueue()
        open_list.insert(self.node_start)
        while len(open_list.nodes) > 0:
            # pick the best node and remove it from open_list
            node_best = open_list.pop()
            # add best node to closed_list
            closed_list.insert(node_best)
            # check the robot reaches the goal position
            if node_best == self.node_goal:
                self.node_goal.node_p = node_best.node_p
                break
            # get all neighboring nodes based on the robot motion model
            neighbors = self.get_neighboring_nodes(node_best)
            for node_m in neighbors:
                # calculate edge cost between two consecutive nodes
                cost_edge, is_feasible = self.cost_estimator.calc_cost(node_best, node_m)
                if not is_feasible:
                    continue
                # calculate f(m) = (g(n) + cost(n, m)) + h(m)
                node_best.g = node_best.f - node_best.h
                node_m.g = node_best.g + cost_edge
                node_m.h = self.calc_heuristic(node_m)
                node_m.f = node_m.g + node_m.h
                # neighbor is in closed_list
                if closed_list.test(node_m):
                    continue
                # neighbor is in open_list
                if open_list.test(node_m):
                    # update open_list
                    open_list.update(node_m)
                # neighbor is not in both open_list and closed_list
                else:
                    # add node_ into open_list
                    open_list.insert(node_m)
        self.path, self.nodes = self.get_final_path()
        return self.path, self.nodes
                
    def get_neighboring_nodes(self, node):
        """
        get_neighboring_nodes: get neighbors from the current position

        :param node: current node
        """
        neighbors = []
        for motion_ in self.motion:
            x_id = node.xy_ids[0] + motion_[0]
            y_id = node.xy_ids[1] + motion_[1]
            if self.is_inside_map(x_id, y_id):
                node_ = Node((x_id, y_id), node)
                neighbors.append(node_)
        return neighbors

    def is_inside_map(self, x_id: int, y_id: int):
        """
        is_inside_map: verify the given x- and y-axis indices are feasible or not
        
        :param x_id: x-axis index
        :param y_id: y-axis index
        """
        if 0 <= x_id < self.map.n and 0 <= y_id < self.map.n:
            return True
        else:
            return False

    def calc_heuristic(self, node, vel_ref: float = 0.1):
        """
        calc_heuristic: calculate heuristic cost

        :param node: node class
        :param vel_ref: reference velocity (0.1 m/sec)
        """
        pos = self.calc_pos_from_xy_id(node.xy_ids)
        pos_goal = self.calc_pos_from_xy_id(self.node_goal.xy_ids)
        cost = np.linalg.norm(pos - pos_goal) / vel_ref
        return cost

    def calc_pos_from_xy_id(self, xy_ids: tuple):
        """
        calc_pos_from_xy_id: calculate positional information from indices

        :param xy_ids: x- and y-axis indices
        """
        xy_pos = np.array(xy_ids) * self.map.res + np.array([self.map.lower_left_x, self.map.lower_left_y])
        z_pos = self.map.get_value_from_xy_id(xy_ids[0], xy_ids[1])
        pos = np.append(xy_pos, z_pos)
        return pos

    def get_final_path(self):
        """
        get_final_path: get final path
        
        """
        if self.node_goal.node_p is None: # no solution found
            return None, []
        path = np.empty((0, 3), float)
        node_ = self.node_goal
        nodes = []
        while node_ is not None:
            nodes.append(node_)
            path = np.vstack([path, self.calc_pos_from_xy_id(node_.xy_ids)])
            node_ = node_.node_p
        nodes.reverse()
        return path, nodes

    def set_final_path(self, nodes: list):
        """
        set_final_path: set final path

        :param nodes: solution of path 
        """
        if not nodes:
            path = None
        else:
            path = np.empty((0, 3), float)
            for node_c in nodes:
                path = np.vstack([path, self.calc_pos_from_xy_id(node_c.xy_ids)])
        self.path, self.nodes = path, nodes

    def execute_final_path(self):
        """
        execute_final_path: execute final path following

        """
        if self.path is None:
            metrics = Metrics(is_solved=False, is_feasible=False)
            return metrics
        # init metrics
        dist = 0
        obs_time = 0 # time
        est_cost = 0 # risk-associated cost
        slips = []
        is_feasible = True
        total_traj = np.empty((0, 3), float)
        time_slips = np.empty((0, 2), float)
        for i, node_c in enumerate(self.nodes):
            if node_c == self.node_goal:
                break
            node_n = self.nodes[i+1]
            # get edge wise information
            # distance
            dist_edge, _, _ = self.cost_observer.get_edge_information(node_c, node_n)
            # estimation (planning) metrics
            est_cost_edge, _ = self.cost_estimator.calc_cost(node_c, node_n, metric="ra-time")
            # observation (execution) metrics
            obs_time_edge, is_feasible_edge, slips_edge = self.cost_observer.calc_cost(node_c, node_n)
            # path execution failed
            if not is_feasible_edge:
                self.node_failed = node_n
                obs_time = None
                is_feasible = False
                total_traj = np.vstack([total_traj, self.calc_pos_from_xy_id(node_c.xy_ids)])
                if max(slips_edge) == 1:
                    s = max(slips_edge)
                elif min(slips_edge) == -1:
                    s = min(slips_edge)
                if i != 0:
                    time_slip = np.array([[time_slips[-1, 0] + 1, s]])
                else:
                    time_slip = np.array([0, s])
                time_slips = np.vstack([time_slips, time_slip])                
            # increment total info.
            dist += dist_edge
            est_cost += est_cost_edge
            if self.node_failed is None: # only when observation is possible
                obs_time += obs_time_edge
                traj, time_slip = self.generate_traj(node_c, node_n, dist_edge / obs_time_edge)
                total_traj = np.vstack([total_traj, traj])
                if i != 0:
                    time_slip[:, 0] += time_slips[-1, 0] + 1
                time_slips = np.vstack([time_slips, time_slip])
            slips.append(max(slips_edge))
            if not is_feasible:
                break
        # add calculated information into metrics structure
        metrics = Metrics(path=self.path,
                        dist=dist,
                        obs_time=obs_time,
                        est_cost=est_cost,
                        max_slip=max(slips),
                        is_solved=True,
                        is_feasible=is_feasible,
                        node_failed=self.node_failed,
                        total_traj=total_traj,
                        time_slips=time_slips)
        return metrics

    def generate_traj(self, node_c, node_n, vel):
        """
        generate_traj: generate trajectory for given edge traverse
        """
        # get positional information
        pos_c = self.calc_pos_from_xy_id(node_c.xy_ids)
        pos_n = self.calc_pos_from_xy_id(node_n.xy_ids)
        # get vector
        r_dir = pos_n - pos_c
        r_uni = r_dir / np.linalg.norm(r_dir)
        # define steps
        steps = int(np.linalg.norm(r_dir) / vel)
        vels = np.full(steps, vel)
        dr_vecs = r_uni[np.newaxis, :] * vels[:, np.newaxis]
        r_vecs = np.cumsum(dr_vecs, axis=0)
        # add first node
        traj = pos_c + r_vecs
        traj = np.vstack([pos_c, traj])
        if vel <= 0.1:
            s = (0.1 - vel) / 0.1
        else:
            s = (0.1 - vel) / vel
        slips = np.full(steps, s)
        times = np.arange(steps)
        time_slip = np.stack((times, slips), axis=1)
        return traj, time_slip

    def plot_envs(self, figsize: tuple = (18, 8), is_tf: bool = True):
        """
        plot_envs: plot 2D and 2.5 D figures and terrain classification results

        :param figsize: size of figure
        :param is_tf: existence of terrain features
        """
        sns.set()
        sns.set_style('whitegrid')
        fig = plt.figure(figsize=figsize)
        fig.suptitle("Terrain classification and path planning results")
        # plot 2.5 and 2d envs
        _, _ = self.map.plot_3d_map(fig=fig, rc=245, is_tf=is_tf)
        _, ax = self.map.plot_2d_map(fig=fig, rc=144)
        # plot actual and predicted models
        self.smg.visualize(fig=fig, rc_ax1=246, rc_ax2=247)
        return fig, ax

    def plot_terrain_classification(self, fig):
        """
        plot_terrain_classification: show terrain classification results

        :param fig: figure
        """
        visualize(vmin=0, vmax=9, n_row=2, n_col=4, fig=fig, 
        terrain_texture_map=self.map.data.color, 
        ground_truth=np.reshape(self.map.data.t_class, (self.map.n, self.map.n)), 
        prediction=np.reshape(self.pred, (self.map.n, self.map.n)))

    def plot_final_path(self, ax, metrics, color: str = "black", plan_type: str = "optimal planner"):
        """
        plot_final_path: plot final path

        :param ax: 
        :param metrics: metrics containing planning results
        :param color: color of path
        :param plan_type: type of planner
        """
        # visualize planned path
        if metrics.path is not None:
            if metrics.node_failed is None:
                ax.plot(metrics.path[:, 0], metrics.path[:, 1], linewidth=4, color=color, 
                        label="%s, est. cost: %.2f, obs. time: %.2f [min], max. slip: %.2f" % (plan_type, metrics.est_cost / 60, metrics.obs_time / 60, metrics.max_slip))
            else:
                ax.plot(metrics.path[:, 0], metrics.path[:, 1], linewidth=4, color=color, 
                        label='%s, failed trial' % (plan_type))
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), fontsize=8)
        # visualize start/goal position
        xy_pos_start = self.calc_pos_from_xy_id(self.node_start.xy_ids)
        xy_pos_goal = self.calc_pos_from_xy_id(self.node_goal.xy_ids)
        ax.plot(xy_pos_start[0], xy_pos_start[1], marker="s", markersize=6, markerfacecolor="blue", markeredgecolor="black")
        ax.plot(xy_pos_goal[0], xy_pos_goal[1], marker="*", markersize=12, markerfacecolor="yellow", markeredgecolor="black")
        if metrics.node_failed is not None:
            xy_pos_failed = self.calc_pos_from_xy_id(metrics.node_failed.xy_ids)
            ax.plot(xy_pos_failed[0], xy_pos_failed[1], marker="X", markersize=9, markerfacecolor="red", markeredgecolor="black")
