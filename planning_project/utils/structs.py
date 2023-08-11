"""
description: a script containing data structures
author: Masafumi Endo
"""

import dataclasses
import numpy as np

@dataclasses.dataclass
class PlanMetrics:
    """
    structure containing types of planning algorithms

    :param is_plan: whether the planner do planning or not
    :param type_model: slip model name
    :param type_embed: uncertainty embed method
    :param alpha: alpha value controls var/cvar behavior
    """
    is_plan: bool 
    type_model: str
    type_embed: str
    alpha: float = None

@dataclasses.dataclass
class AbsEval:
    """
    structure containing evaluated metrics

    """
    plan_metrics: any
    results: any
    is_solved: float = None
    is_feasible: float = None
    # mean and var
    dist: np.array = None
    est_cost: np.array = None
    obs_time: np.array = None
    max_slip: np.array = None

@dataclasses.dataclass
class HyperParams:
    """
    structure containing hyper parameters

    :param nn_model_dir: directory to neural network model
    :param data_dir: directory to slip model
    :param results_dir: directory to save results

    :param n_terrains: number of terrain types
    :param res: resolution of map environment
    
    :param start_pos: start position for path planning
    :param goal_pos: goal position for path planning 

    :param plan_metrics: parameter for specifying planning metrics
    :param is_plan: operator for planning or not

    :param idx_dataset: dataset index
    :param idx_instances: instance index
    """
    # param for directory
    nn_model_dir: str
    data_dir: str
    results_dir: str
    # param for map
    n_terrains: int
    res: float
    # param for planning
    start_pos: tuple
    goal_pos: tuple
    # param for learning-based planner 
    plan_metrics: list
    # param for experiment
    idx_dataset: int
    idx_instances: list

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

@dataclasses.dataclass
class EvalMetrics:
    """
    structure containing path planning evaluation metrics

    :param path: final path derived by planner
    :param dist: total distance when path execution
    :param time: total time when path execution
    :param est_cost: esimtated risk-associated time when path planning
    :param obs_cost: observed risk-associated time when path execution
    :param max_slip: max. slip ratio when path execution
    :param is_solved: check the problem is solved or not
    :param is_feasible: feasibility check for path execution
    :param node_failed: failed node (positional info. in node class)
    """
    path: np.array = None
    dist: float = None
    obs_time: float = None
    est_cost: float = None
    max_slip: float = None
    is_solved: bool = True
    is_feasible: bool = True
    node_failed: any = None
    total_traj: np.array = None
    time_slips: np.array = None

@dataclasses.dataclass
class HyperParams:
    """
    structure containing hyper parameters

    :param data_dir: directory to dataset
    :param nn_model_dir: direcotry to neural network model
    :param log_dir: directory to logger
    :param lr: learning rate
    :param th: metrics threshold
    :param batch_size_train: batch size for training 
    :param num_epochs: number of epochs
    :param epoch_lr: number of epochs that decreases learning rate
    :param patience: early stopping condition
    :param max_score: max score
    """
    data_dir: str
    nn_model_dir: str
    log_dir: str

    lr: float = 1e-3
    th: float = 0.5
    batch_size_train: int = 8
    batch_size_valid: int = 1
    num_epochs: int = 50
    epoch_lr: int = 10
    patience: int = 5
    max_score: float = 0