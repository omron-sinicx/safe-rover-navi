"""
description: visualize 3d map environment w/ generated trajectory
author: Masafumi Endo
"""

import pickle
import sys, os
BASE_PATH = os.path.dirname(__file__)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pylab
import seaborn as sns

from planning_project.env.env import GridMap
from planning_project.env.slip_models import SlipModel
from planning_project.utils.data import DataSet, create_int_label
from scripts.eval import Runner 
from scripts.eval import PlanMetrics
from scripts.eval import HyperParams

COLOR = ['red', 'blue', 'green', 'lime', 'aqua', 'magenta']
plt.rcParams['figure.subplot.bottom'] = 0.15

def backtrack(dataset, name_instance):
    for i in range(len(dataset)):
        name_instance_ = os.path.splitext(dataset.ids[i])[0]
        if name_instance_ == name_instance:
            idx_instance = i
    return idx_instance

def plot_2d_graph(fig = None, rc: int=111, steps = None):
    sns.set()
    sns.set_style('whitegrid')
    if fig is None:
        fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(rc)
    ax.set_xlabel("time min")
    ax.set_ylabel("slip %")
    ax.set_xlim(0, steps)
    ax.set_ylim(-1, 1)
    ax.set_xticks([x for x in range(0, steps + 180, 180)])
    ax.set_yticks([-1, 0, 1])
    xticks, _ = pylab.xticks()
    pylab.xticks(xticks, ["%d" % x for x in xticks / 60])
    yticks, _ = pylab.yticks()
    pylab.yticks(yticks, ["%d" % y for y in 100 * yticks])
    return ax

def plot_3d_map(grid_map, fig = None, rc: int = 111):
    xx, yy = np.meshgrid(np.arange(0.0, grid_map.n * grid_map.res, grid_map.res),
                            np.arange(0.0, grid_map.n * grid_map.res, grid_map.res))

    grid_data = np.reshape(grid_map.data.height, (grid_map.n, grid_map.n))
    data = grid_map.data.height

    sns.set()
    sns.set_style('whitegrid')
    if fig is None:
        fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(rc, projection="3d")
    hmap = ax.plot_surface(xx + grid_map.res / 2.0, yy + grid_map.res / 2.0, grid_data, 
                        facecolors=grid_map.data.color, linewidth=0, antialiased=False)
    ax.set_xlabel("x-axis m")
    ax.set_ylabel("y-axis m")
    ax.set_zticks(np.arange(xx.min(), xx.max(), 10))
    ax.view_init(elev=40, azim=-45)
    ax.set_box_aspect((1, 1, 0.2))
    ax.set_xlim(xx.min(), xx.max() + grid_map.res)
    ax.set_ylim(yy.min(), yy.max() + grid_map.res)
    ax.set_zlim(min(data), xx.max() / 10)

    return hmap, ax

def calc_pos_from_xy_id(grid_map, xy_ids: tuple):
    xy_pos = np.array(xy_ids) * grid_map.res + np.array([grid_map.lower_left_x, grid_map.lower_left_y])
    z_pos = grid_map.get_value_from_xy_id(xy_ids[0], xy_ids[1])
    pos = np.append(xy_pos, z_pos)
    return pos

def plot_static_paths(ax, metrics, hyper_params, grid_map):
    pos_start = hyper_params.start_pos
    x_id, y_id = grid_map.get_xy_id_from_xy_pos(pos_start[0], pos_start[1])
    pos_start = calc_pos_from_xy_id(grid_map, [x_id, y_id])
    pos_goal = hyper_params.goal_pos
    x_id, y_id = grid_map.get_xy_id_from_xy_pos(pos_goal[0], pos_goal[1])
    pos_goal = calc_pos_from_xy_id(grid_map, [x_id, y_id])
    for i in range(len(metrics)):
        metrics_ = metrics[i][0]
        if metrics_.path is not None:
            ax.plot(metrics_.path[:, 0], metrics_.path[:, 1], metrics_.path[:, 2],
            linewidth=4, color=COLOR[i])
        else:
            pass
        if metrics_.node_failed is not None:
            xy_pos_failed = calc_pos_from_xy_id(grid_map, metrics_.node_failed.xy_ids)
            ax.plot(xy_pos_failed[0], xy_pos_failed[1], xy_pos_failed[2], marker="X", markersize=10, markerfacecolor="red", markeredgecolor="black")
    ax.plot(pos_start[0], pos_start[1], pos_start[2], marker="s", markersize=10, markerfacecolor="blue", markeredgecolor="black")
    ax.plot(pos_goal[0], pos_goal[1], pos_goal[2], marker="*", markersize=15, markerfacecolor="yellow", markeredgecolor="black")

def plot_animated_paths(metrics, hyper_params, grid_map):
    fig_tj = plt.figure(figsize=(8, 6))
    _, ax = plot_3d_map(grid_map, fig=fig_tj)
    # start and goal settings
    pos_start = hyper_params.start_pos
    x_id, y_id = grid_map.get_xy_id_from_xy_pos(pos_start[0], pos_start[1])
    pos_start = calc_pos_from_xy_id(grid_map, [x_id, y_id])
    pos_goal = hyper_params.goal_pos
    x_id, y_id = grid_map.get_xy_id_from_xy_pos(pos_goal[0], pos_goal[1])
    pos_goal = calc_pos_from_xy_id(grid_map, [x_id, y_id])
    ax.plot(pos_start[0], pos_start[1], pos_start[2], marker="s", markersize=15, markerfacecolor="blue", markeredgecolor="black")
    ax.plot(pos_goal[0], pos_goal[1], pos_goal[2], marker="*", markersize=20, markerfacecolor="yellow", markeredgecolor="black")

    # animation section
    steps = 0
    for i in range(len(metrics)):
        metrics_ = metrics[i][0]
        steps_ = len(metrics_.total_traj[:, 0])
        if steps_ > steps:
            steps = steps_
    fig_th = plt.figure(figsize=(10, 4))
    ax_th = plot_2d_graph(fig=fig_th, steps=steps)
    ims_tj = []
    ims_th = []
    for t in range(steps):
        if t % 5 == 0 or t == steps - 1:
            im_traj0, im_robo0 = get_ims_per_traj(ax, metrics[0][0], t, COLOR[0])
            im_traj1, im_robo1 = get_ims_per_traj(ax, metrics[1][0], t, COLOR[1])
            im_traj2, im_robo2 = get_ims_per_traj(ax, metrics[2][0], t, COLOR[2])
            im_traj3, im_robo3 = get_ims_per_traj(ax, metrics[3][0], t, COLOR[3])
            im_traj4, im_robo4 = get_ims_per_traj(ax, metrics[4][0], t, COLOR[4])
            im_traj5, im_robo5 = get_ims_per_traj(ax, metrics[5][0], t, COLOR[5])
            ims_tj.append(im_traj0 + im_robo0 + 
                       im_traj1 + im_robo1 +
                       im_traj2 + im_robo2 + 
                       im_traj3 + im_robo3 + 
                       im_traj4 + im_robo4 +
                       im_traj5 + im_robo5)
            im_traj0, im_robo0 = get_ims_per_slip(ax_th, metrics[0][0], t, COLOR[0])
            im_traj1, im_robo1 = get_ims_per_slip(ax_th, metrics[1][0], t, COLOR[1])
            im_traj2, im_robo2 = get_ims_per_slip(ax_th, metrics[2][0], t, COLOR[2])
            im_traj3, im_robo3 = get_ims_per_slip(ax_th, metrics[3][0], t, COLOR[3])
            im_traj4, im_robo4 = get_ims_per_slip(ax_th, metrics[4][0], t, COLOR[4])
            im_traj5, im_robo5 = get_ims_per_slip(ax_th, metrics[5][0], t, COLOR[5])
            ims_th.append(im_traj0 + im_robo0 + 
                       im_traj1 + im_robo1 +
                       im_traj2 + im_robo2 + 
                       im_traj3 + im_robo3 + 
                       im_traj4 + im_robo4 +
                       im_traj5 + im_robo5)
        else:
            continue
    return ims_tj, fig_tj, ims_th, fig_th

def get_ims_per_slip(ax, metrics, t, color):
    if t < len(metrics.time_slips[:, 0]) - 1:
        traj = metrics.time_slips[0:t+1, :]
        robo = traj[-1, :]
        im_traj = ax.plot(traj[:, 0], traj[:, 1], linewidth=4, color=color)
        im_robo = ax.plot(robo[0], robo[1], marker='o', color=color)
    else:
        traj = metrics.time_slips
        robo = traj[-1, :]
        im_traj = ax.plot(traj[:, 0], traj[:, 1], linewidth=4, color=color)
        im_robo = ax.plot(robo[0], robo[1], marker='o', color=color)
        if not metrics.is_feasible:
            im_robo = ax.plot(robo[0], robo[1], marker="X", markersize=15, markerfacecolor=color, markeredgecolor="black")
    return im_traj, im_robo

def get_ims_per_traj(ax, metrics, t, color):
    if t < len(metrics.total_traj[:, 0]) - 1:
        traj = metrics.total_traj[0:t+1, :]
        robo = traj[-1, :]
        im_traj = ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=4, color=color)
        im_robo = ax.plot(robo[0], robo[1], robo[2], marker='o', color=color)
    else:
        traj = metrics.total_traj
        robo = traj[-1, :]
        im_traj = ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=4, color=color)
        im_robo = ax.plot(robo[0], robo[1], robo[2], marker='o', color=color)
        if not metrics.is_feasible:
            im_robo = ax.plot(robo[0], robo[1], robo[2], marker="X", markersize=15, markerfacecolor=color, markeredgecolor="black")
    return im_traj, im_robo
    
def set_env(hyper_params):
    dataset = DataSet(os.path.join(hyper_params.data_dir, 'data%02d/' % (hyper_params.idx_dataset)), "test")
    color_map, mask = dataset[hyper_params.idx_instances[0]]
    mask = create_int_label(dataset.to_image(mask))
    height_map = dataset.get_height_map(hyper_params.idx_instances[0])
    grid_map = GridMap(color_map.shape[1], hyper_params.res)
    grid_map.load_env(height_map, mask, dataset.to_image(color_map))
    grid_map.data.name_instance = os.path.splitext(dataset.ids[hyper_params.idx_instances[0]])[0]
    return grid_map

def set_hyper_params(idx_dataset, n_terrains, idx_instance, plan_metrics):
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
                            idx_instances=[idx_instance])
    return hyper_params

def main():
    idx_dataset = 3
    n_terrains = 8

    data_dir = os.path.join(BASE_PATH, '../datasets/data%02d/' % (idx_dataset))
    dataset = DataSet(data_dir, "test")
    name_instance = 'env_07_0009'
    idx_instance = backtrack(dataset, name_instance)
    is_plan = False
    plan_metrics = [
                PlanMetrics(is_plan=is_plan, type_model="gsm", type_embed="mean", alpha=None),
                PlanMetrics(is_plan=is_plan, type_model="gsm", type_embed="var", alpha=0.99),
                PlanMetrics(is_plan=is_plan, type_model="gsm", type_embed="cvar", alpha=0.99),
                PlanMetrics(is_plan=is_plan, type_model="gmm", type_embed="mean", alpha=None),
                PlanMetrics(is_plan=is_plan, type_model="gmm", type_embed="var", alpha=0.99),
                PlanMetrics(is_plan=is_plan, type_model="gmm", type_embed="cvar", alpha=0.99),
                ] # proposed method
    hyper_params = set_hyper_params(idx_dataset, n_terrains, idx_instance, plan_metrics)
    runner = Runner(hyper_params, is_run=True)
    metrics = runner.metrics_ds
    
    grid_map = set_env(hyper_params)
    ims_tj, fig_tj, ims_th, fig_th = plot_animated_paths(metrics, hyper_params, grid_map)
    anm = animation.ArtistAnimation(fig_tj, ims_tj, interval=30)
    name_saved = os.path.join(hyper_params.results_dir, 'data%02d/animations/trajectory/%s.mp4' % (idx_dataset, grid_map.data.name_instance))
    anm.save(name_saved, writer='ffmpeg')
    anm = animation.ArtistAnimation(fig_th, ims_th, interval=30)
    name_saved = os.path.join(hyper_params.results_dir, 'data%02d/animations/time_slip/%s.mp4' % (idx_dataset, grid_map.data.name_instance))
    anm.save(name_saved, writer='ffmpeg')
    plt.show()
    
if __name__ == '__main__':
    main()