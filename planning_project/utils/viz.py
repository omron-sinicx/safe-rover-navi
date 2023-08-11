"""
description: methods to visualize information 
author: Masafumi Endo
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

def viz_terrain_props(vmin: float, vmax: float, n_row: int = 1, n_col: int = None, fig: plt.figure = None, **images):
    """
    viz_terrain_props: visualize input terrain property information in one row

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
        if name == 'ground_truth' or name == 'prediction':
            ax.imshow(image, vmin=vmin, vmax=vmax, cmap="jet")
        elif name == 'likelihood':
            ax.imshow(image, vmin=0, vmax=1, cmap="jet")
        else:
            ax.imshow(image)

def viz_maps(figsize: tuple = (10, 4), is_tf: bool = True):
    """
    viz_maps: visualize 2D and 2.5D figures with given size

    :param figsize: size of figure
    :param is_tf: existence of terrain features
    """
    sns.set()
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=figsize)
    _, ax_3d = viz_3d_map(fig=fig, rc=121, is_tf=is_tf)
    _, ax_2d = viz_2d_map(fig=fig, rc=122)
    plt.tight_layout()
    return ax_2d, ax_3d

def viz_2d_map(grid_map, fig: plt.figure = None, rc: int = 111, field_name: str = "height", 
                title: str = "2D celestial terrain", cmap: str = "viridis", label: str = "height m"):
    """
    viz_2d_map: visualize 2D grid map

    :param grid_data: data to visualize
    :param fig: figure
    :param rc: position specification as rows and columns
    :param i_tf: index of terrain features
    :param title: title of shown figure
    """
    xx, yy = np.meshgrid(np.arange(0.0, grid_map.n * grid_map.res, grid_map.res),
                            np.arange(0.0, grid_map.n * grid_map.res, grid_map.res))

    grid_data = np.reshape(getattr(grid_map.data, field_name), (grid_map.n, grid_map.n))
    data = getattr(grid_map.data, field_name)
    if not fig:
        fig = plt.figure()

    ax = fig.add_subplot(rc)
    hmap = ax.pcolormesh(xx + grid_map.res / 2.0, yy + grid_map.res / 2.0, grid_data,
                            cmap=cmap, vmin=min(data), vmax=max(data))
    ax.set_xlabel("x-axis m")
    ax.set_ylabel("y-axis m")
    ax.set_aspect("equal")
    ax.set_xlim(xx.min(), xx.max() + grid_map.res)
    ax.set_ylim(yy.min(), yy.max() + grid_map.res)
    ax.set_title(title)

    plt.colorbar(hmap, ax=ax, label=label, orientation='horizontal')

    return hmap, ax

def viz_3d_map(grid_map: np.ndarray = None, fig: plt.figure = None, rc: int = 111, is_tf: bool = False):
    """
    viz_3d_map: visualize 2.5D grid map

    :param grid_data: data to visualize
    :param fig: figure
    :param rc: position specification as rows and columns
    :param is_tf: existence of terrain features
    """
    xx, yy = np.meshgrid(np.arange(0.0, grid_map.n * grid_map.res, grid_map.res),
                            np.arange(0.0, grid_map.n * grid_map.res, grid_map.res))

    grid_data = np.reshape(grid_map.data.height, (grid_map.n, grid_map.n))
    data = grid_map.data.height
    if not fig:
        fig = plt.figure()

    ax = fig.add_subplot(rc, projection="3d")
    if not is_tf:
        hmap = ax.plot_surface(xx + grid_map.res / 2.0, yy + grid_map.res / 2.0, grid_data,
                            cmap="viridis", vmin=min(data), vmax=max(data), linewidth=0, antialiased=False)
    else:
        hmap = ax.plot_surface(xx + grid_map.res / 2.0, yy + grid_map.res / 2.0, grid_data, 
                            facecolors=grid_map.data.color, linewidth=0, antialiased=False)
    ax.set_xlabel("x-axis m")
    ax.set_ylabel("y-axis m")
    ax.set_zticks(np.arange(xx.min(), xx.max(), 10))
    ax.view_init(elev=30, azim=45)
    ax.set_box_aspect((1, 1, 0.2))
    ax.set_xlim(xx.min(), xx.max() + grid_map.res)
    ax.set_ylim(yy.min(), yy.max() + grid_map.res)
    ax.set_zlim(min(data), xx.max() / 10)
    ax.set_title("2.5D celestial terrain")

    return hmap, ax


def viz_slip_models(smg, fig: plt.figure = None, rc_ax1: int = 121, rc_ax2: int = 122):
    """
    viz_slip_models: visualize actual and predicted slip models

    :param rc_ax1: position of plot (actual model)
    :param rc_ax2: position of plot (predicted model)
    """

    sns.set()
    if not fig:
        fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(rc_ax1)
    ax2 = fig.add_subplot(rc_ax2)
    color = cm.jet(np.linspace(0, 1, smg.n_terrains))
    for i in range(smg.n_terrains):
        theta = np.linspace(-40, 40, 100)
        model_actual = np.zeros(theta.shape[0])
        for j, theta_ in enumerate(theta):
            model_actual[j] = smg.slip_models[i].latent_model(theta=theta_)
        ax1.plot(theta, model_actual, color=color[i])
        smg.gp_models[i].plot_mean(ax=ax2, color=color[i], label=None)
        smg.gp_models[i].plot_confidence(ax=ax2, color=color[i], label=None)
    ax1.set_title('actual slip model')
    ax1.set_xlim([-20, 20])
    ax1.set_ylim([-1, 1])
    ax1.set_xticks([-20, -10, 0, 10, 20])
    ax1.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax1.set_xlabel('slope angle deg')
    ax1.set_ylabel('slip ratio')
    ax2.set_title('GP prediction of slip model')
    ax2.set_xlim([-20, 20])
    ax2.set_ylim([-1, 1])
    ax2.set_xticks([-20, -10, 0, 10, 20])
    ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax2.set_xlabel('slope angle deg')
    ax2.set_ylabel('slip ratio')