"""
description: evaluation script for learning-based path planning algorithm
author: Masafumi Endo
"""

import dataclasses
import pickle
import time
import sys, os
BASE_PATH = os.path.dirname(__file__)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from planning_project.env.env import GridMap
from planning_project.env.slip_models import SlipModel, SlipModelsGenerator
from planning_project.env.utils import SlipDistributionMap
from planning_project.planner.planner import AStarPlanner, Node
from planning_project.utils.data import DataSet, create_int_label

PATH_COLOR = ['k', 'g', 'b', 'c']

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

class Runner:

    def __init__(self, hyper_params, is_run: bool):
        """
        __init__:
        
        """
        self.hyper_params = hyper_params
        self.is_run = is_run

        self.dataset = None
        self.smg = None
        self.planner = None

        self.metrics_ds = []
        self.results_dir = os.path.join(self.hyper_params.results_dir, 'data%02d/' % (self.hyper_params.idx_dataset))

        self.run_load_experiments()

    def run_load_experiments(self):
        """
        run_load_experiments: run or load experiments

        :param is_run: run planning algorithm when is_run = True
        """
        if self.is_run:
            # generate metrics by running algorithms
            _ = self.run_experiments_for_dataset()
        else:
            # load already generated metrics
            _ = self.load_experiments()

    def absolute_evaluations(self):
        """
        absolute_evaluations: absolultely evaluate path planning performance for dataset

        """
        abs_evals = []
        for plan_metrics, results in zip(self.hyper_params.plan_metrics, self.metrics_ds):
            abs_eval = AbsEval(plan_metrics=plan_metrics, results=results)
            abs_eval = self.__absolute_evaluation(abs_eval)
            abs_evals.append(abs_eval)
        return abs_evals

    def __absolute_evaluation(self, abs_eval):
        """
        __absolute_evaluation: absolutely evaluate path planning performance per metric

        :param plan_metric_id: plan metric index
        """
        # get resulting metrics
        results = abs_eval.results
        is_solved = self.get_field_vals(results, field_name='is_solved')
        is_feasible = self.get_field_vals(results, field_name='is_feasible')
        dist = self.get_field_vals(results, field_name='dist')
        est_cost = self.get_field_vals(results, field_name='est_cost')
        obs_time = self.get_field_vals(results, field_name='obs_time')
        max_slip = self.get_field_vals(results, field_name='max_slip')
        # calculate success rate
        is_solved = np.sum(is_solved) / len(is_solved) * 100
        is_feasible = np.sum(is_feasible) / len(is_feasible) * 100
        # calculate mean/var
        est_cost = est_cost / 60
        obs_time = obs_time / 60
        max_slip = max_slip * 100
        dist_m, dist_v = np.nanmean(dist), np.nanstd(dist) * 2
        est_cost_m, est_cost_v = np.nanmean(est_cost), np.nanstd(est_cost) * 2
        obs_time_m, obs_time_v = np.nanmean(obs_time), np.nanstd(obs_time) * 2
        max_slip_m, max_slip_v = np.nanmean(max_slip), np.nanstd(max_slip) * 2
        abs_eval.is_solved = is_solved
        abs_eval.is_feasible = is_feasible
        abs_eval.dist = np.array([round(dist_m, 1), round(dist_v, 1)])
        abs_eval.est_cost = np.array([round(est_cost_m, 1), round(est_cost_v, 1)])
        abs_eval.obs_time = np.array([round(obs_time_m, 1), round(obs_time_v, 1)])
        abs_eval.max_slip = np.array([round(max_slip_m, 1), round(max_slip_v, 1)])
        return abs_eval

    def run_experiments_for_dataset(self):
        """
        run_experiments_for_dataset: run experiment for given dataset

        """
        self.dataset = DataSet(os.path.join(self.hyper_params.data_dir, 'data%02d/' % (self.hyper_params.idx_dataset)), "test")
        self.smg = SlipModelsGenerator(os.path.join(self.hyper_params.data_dir, 'data%02d/' % (self.hyper_params.idx_dataset)), self.hyper_params.n_terrains)
        self.planner = AStarPlanner(map=None, smg=self.smg, hyper_params=self.hyper_params)
        idx_instances = self.hyper_params.idx_instances
        if not idx_instances:
            idx_instances = list(range(len(self.dataset)))
        grid_maps = []
        for idx_instance in idx_instances:
            grid_map = self.set_test_env(idx_instance)
            grid_maps.append(grid_map)
        for plan_metrics in self.hyper_params.plan_metrics:
            metrics_env = self.run_expriments_for_metrics(grid_maps, plan_metrics)
            self.metrics_ds.append(metrics_env)
        return self.metrics_ds

    def run_expriments_for_metrics(self, grid_maps, plan_metrics):
        """
        run_experiments_for_metrics: run experiments for datasets with given metrics

        :param idx_instance: index instances
        :param plan_metrics: specified metrics for planning
        """
        metrics_envs = Parallel(n_jobs=1)(delayed(self.run_experiment_for_env)(grid_map, plan_metrics) for grid_map in grid_maps)
        return metrics_envs

    def load_experiments(self):
        """
        load_experiments: load experiments' results

        """
        self.dataset = DataSet(os.path.join(self.hyper_params.data_dir, 'data%02d/' % (self.hyper_params.idx_dataset)), "test")
        idx_instances = self.hyper_params.idx_instances
        if not idx_instances:
            idx_instances = list(range(len(self.dataset)))
        for plan_metrics in self.hyper_params.plan_metrics:
            metrics_env = []
            for idx_instance in idx_instances:
                name_instance = os.path.splitext(self.dataset.ids[idx_instance])[0] 
                print("load experiment for instance ", name_instance)
                if plan_metrics.type_model == "gmm" and plan_metrics.type_embed == "cvar":
                    with open(os.path.join(self.results_dir, 'metrics/%s_%s_%s/%s.pkl' % (plan_metrics.type_model, plan_metrics.type_embed, str(int(plan_metrics.alpha * 100)), name_instance)), mode='rb') as f:
                        metrics = pickle.load(f)
                else:
                    with open(os.path.join(self.results_dir, 'metrics/%s_%s/%s.pkl' % (plan_metrics.type_model, plan_metrics.type_embed, name_instance)), mode='rb') as f:
                        metrics = pickle.load(f)
                metrics_env.append(metrics)
            self.metrics_ds.append(metrics_env)
        return self.metrics_ds

    def run_experiment_for_env(self, grid_map, plan_metrics):
        """
        run_experiment_for_env: run experiment for given map environment by specific planning algorithm

        :param idx_instance: index of environment instance
        """
        print('start to run experiment for the conditions of : %s %s %s' %(plan_metrics.type_model, plan_metrics.type_embed, grid_map.data.name_instance))
        metrics = self.execute_planner(grid_map, grid_map.data.name_instance, plan_metrics)
        if plan_metrics.type_model == "gmm" and plan_metrics.type_embed == "cvar":
            with open(os.path.join(self.results_dir, 'metrics/%s_%s_%s/%s.pkl' % (plan_metrics.type_model, plan_metrics.type_embed, str(int(plan_metrics.alpha * 100)), grid_map.data.name_instance)), mode='wb') as f:
                pickle.dump(metrics, f)
        else:
            with open(os.path.join(self.results_dir, 'metrics/%s_%s/%s.pkl' % (plan_metrics.type_model, plan_metrics.type_embed, grid_map.data.name_instance)), mode='wb') as f:
                pickle.dump(metrics, f)
        # show and save figure
        fig = self.visualize(planning_results=metrics)
        if plan_metrics.type_model == "gmm" and plan_metrics.type_embed == "cvar":
            fig.savefig(os.path.join(self.results_dir, 'figs/%s_%s_%s/img/%s.png' % (plan_metrics.type_model, plan_metrics.type_embed, str(int(plan_metrics.alpha * 100)), grid_map.data.name_instance)))
            with open(os.path.join(self.results_dir, 'figs/%s_%s_%s/obj/%s.pkl' % (plan_metrics.type_model, plan_metrics.type_embed, str(int(plan_metrics.alpha * 100)), grid_map.data.name_instance)), mode='wb') as f:
                pickle.dump(fig, f)
        else:
            fig.savefig(os.path.join(self.results_dir, 'figs/%s_%s/img/%s.png' % (plan_metrics.type_model, plan_metrics.type_embed, grid_map.data.name_instance)))
            with open(os.path.join(self.results_dir, 'figs/%s_%s/obj/%s.pkl' % (plan_metrics.type_model, plan_metrics.type_embed, grid_map.data.name_instance)), mode='wb') as f:
                pickle.dump(fig, f)
        plt.close()
        return metrics

    def set_test_env(self, idx_instance: int):
        """
        set_test_env: set test environment for planning

        :param idx_instance: index of environment instance
        """
        name_instance = os.path.splitext(self.dataset.ids[idx_instance])[0]
        color_map, mask = self.dataset[idx_instance]
        mask = create_int_label(self.dataset.to_image(mask))
        height_map = self.dataset.get_height_map(idx_instance)

        n = color_map.shape[1]
        res = self.hyper_params.res
        grid_map = GridMap(n, res)
        grid_map.load_env(height_map, mask, self.dataset.to_image(color_map))
        sdm = SlipDistributionMap(grid_map, self.smg)
        grid_map.data.slip = sdm.set_slip()
        grid_map.data.name_instance = name_instance
        return grid_map

    def execute_planner(self, grid_map, name_instance: str, plan_metrics):
        """
        execute_planner: execute assigned path planning algorithm

        :param grid_map: target map object
        :param name_instance: instance name
        :param plan_metrics: structure containing planning metrics
        """
        self.planner.reset(grid_map, 
                        self.hyper_params.start_pos, 
                        self.hyper_params.goal_pos,
                        plan_metrics)
        if plan_metrics.type_model != "gtm":
            _ = self.planner.predict(self.dataset.to_tensor(grid_map.data.color))
        if plan_metrics.is_plan:
            print("start path planning!")
            start = time.time()
            _, nodes = self.planner.search_path()
            run_time = time.time() - start
            print("done path planning! running time...", run_time)
            # save solution
            if plan_metrics.type_model == "gmm" and plan_metrics.type_embed == "cvar":
                with open(os.path.join(self.results_dir, 'solutions/%s_%s_%s/%s.pkl' % 
                (plan_metrics.type_model, plan_metrics.type_embed, str(int(plan_metrics.alpha * 100)), name_instance)), mode='wb') as f:
                    pickle.dump(nodes, f)
            else:
                with open(os.path.join(self.results_dir, 'solutions/%s_%s/%s.pkl' % 
                (plan_metrics.type_model, plan_metrics.type_embed, name_instance)), mode='wb') as f:
                    pickle.dump(nodes, f)
        else:
            # load solution
            print(plan_metrics.type_model, plan_metrics.type_embed)
            if plan_metrics.type_model == "gmm" and plan_metrics.type_embed == "cvar":
                with open(os.path.join(self.results_dir, 'solutions/%s_%s_%s/%s.pkl' % 
                (plan_metrics.type_model, plan_metrics.type_embed, str(int(plan_metrics.alpha * 100)), name_instance)), mode='rb') as f:
                    nodes = pickle.load(f)
            else:
                with open(os.path.join(self.results_dir, 'solutions/%s_%s/%s.pkl' % 
                (plan_metrics.type_model, plan_metrics.type_embed, name_instance)), mode='rb') as f:
                    nodes = pickle.load(f)
            self.planner.set_final_path(nodes)
        metrics = self.planner.execute_final_path()
        if plan_metrics.type_model == "gtm":
            _ = self.planner.predict(self.dataset.to_tensor(grid_map.data.color))
        return metrics

    def get_field_vals(self, results, field_name: str):
        """
        get_field_vals: get field value 
        
        :param field_name: name of field
        :param plan_metric_id: plan metric index
        """
        n_instances = len(results)
        arr_metric = np.zeros((n_instances, 1))
        for idx_instance in range(n_instances):
            metric = getattr(results[idx_instance], field_name)
            if metric is None:
                arr_metric[idx_instance, 0] = np.nan
            else:
                arr_metric[idx_instance, 0] = metric
        return arr_metric

    def visualize(self, **dict_metrics):
        """
        visualize: visualize results including terrain classification, slip prediction, and path planning

        :param dict_metrics: structure containing path planning metrics
        """
        fig, ax = self.planner.plot_envs() # environment map
        self.planner.plot_terrain_classification(fig) # terrain classification results
        for i, (name, metrics) in enumerate(dict_metrics.items()):
            self.planner.plot_final_path(ax,
                                        metrics,
                                        color=PATH_COLOR[i], 
                                        plan_type=(' '.join(name.split('_'))))
        plt.tight_layout()

        return fig

def show_eval(runner, is_abs: bool):
    if is_abs:
        abs_evals = runner.absolute_evaluations()
        for abs_eval in abs_evals:
            print("trav. estimator : ", abs_eval.plan_metrics.type_model, ", risk infer : ", abs_eval.plan_metrics.type_embed)
            print("solved rate : ", abs_eval.is_solved)
            print("success rate : ", abs_eval.is_feasible)
            print("distance : ", abs_eval.dist)
            print("cost : ", abs_eval.est_cost)
            print("time : ", abs_eval.obs_time)
            print("max. slip : ", abs_eval.max_slip)
            print(" ")

def run_experiments_for_metrics(idx_dataset: int, n_terrains: int, idx_instances: list, plan_metrics, is_run: bool):
    # set hyper-params
    # param for directory
    nn_model_dir = os.path.join(BASE_PATH, '../trained_models/models/data%02d/best_model.pth' % (idx_dataset))
    data_dir = os.path.join(BASE_PATH, '../datasets/')
    results_dir = os.path.join(BASE_PATH, '../results/')
    # param for map
    res = 1
    # param for planning
    margin = 8
    start_pos = (margin, margin)
    goal_pos = (96 - margin, 96 - margin)
    hyper_params = HyperParams(nn_model_dir=nn_model_dir,
                            data_dir=data_dir,
                            results_dir=results_dir,
                            n_terrains=n_terrains,
                            res=res,
                            start_pos=start_pos,
                            goal_pos=goal_pos,
                            plan_metrics=plan_metrics,
                            idx_dataset=idx_dataset,
                            idx_instances=idx_instances)
    runner = Runner(hyper_params, is_run=is_run)
    show_eval(runner=runner, is_abs=True)

def main():
    idx_dataset = 2
    n_terrains = 10
    idx_instances = list(range(1))
    is_run = True
    is_plan = True
    plan_metrics = [
                    PlanMetrics(is_plan=is_plan, type_model="gtm", type_embed="mean", alpha=None),
                    PlanMetrics(is_plan=is_plan, type_model="gsm", type_embed="mean", alpha=None),
                    PlanMetrics(is_plan=is_plan, type_model="gsm", type_embed="var", alpha=0.99),
                    PlanMetrics(is_plan=is_plan, type_model="gsm", type_embed="cvar", alpha=0.99),
                    PlanMetrics(is_plan=is_plan, type_model="gmm", type_embed="mean", alpha=None),
                    PlanMetrics(is_plan=is_plan, type_model="gmm", type_embed="var", alpha=0.99),
                    PlanMetrics(is_plan=is_plan, type_model="gmm", type_embed="cvar", alpha=0.99),
                    ] # proposed method
    run_experiments_for_metrics(idx_dataset, n_terrains, idx_instances, plan_metrics, is_run)

if __name__ == '__main__':
    main()