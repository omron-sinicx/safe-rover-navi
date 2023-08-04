"""
description: set SlipModels class
author: Masafumi Endo
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import os
BASE_PATH = os.path.dirname(__file__)

import GPy

class SlipModel:

    def __init__(self, seed: float):
        """
        __init__: 

        :param seed: seed to fix randomness
        """

        self.seed =seed
        self.rng = np.random.default_rng(self.seed)
        # set params for quadratic function
        self.gain = self.rng.uniform(1e-5, 8)
        self.c = self.rng.uniform(0.0, 0.1)
        # set params for observation noise
        self.scale = 0.05

        self.x = None
        self.y = None
        self.model_param = None

    def generate_data(self, x):
        """
        generate_data: generate pairs of slope vs. slip data to train GP models

        :param x: GP input (terrain inclination for training)
        """
        self.x = x
        self.x = np.sort(self.x)
        self.y = np.zeros(self.x.shape[0])
        for i, theta in enumerate(self.x):
            self.y[i] = self.observe_slip(theta)
        return self.x, self.y

    def observe_slip(self, theta: float):
        """
        observe_slip: observation models contaminated by random noise

        :param theta: terrain inclination
        """
        obs = self.latent_model(theta) + self.rng.normal(0, self.scale)
        return obs

    def latent_model(self, theta: float):
        """
        latent_model: actual slip model for each terrain

        :param theta: terrain inclination
        """
        if theta >= 0:
            slip = self.gain * 1e-3 * theta**2 + self.c
        else:
            slip = - self.gain * 1e-3 * theta**2 + self.c
        return slip

    def fit_gpr(self, gp_model):
        """
        fit_gpr: train given gp model with generated data, then store model parameters

        :param gp_model: gaussian process model
        """
        gp_model.optimize()
        self.model_param = gp_model.param_array
        return gp_model

    def load_gpr(self, gp_model):
        """
        load_gpr: load the model parameters into given gp model

        :param gp_model: gaussian process model
        """
        gp_model.update_model(False)
        gp_model.initialize_parameter()
        gp_model[:] = self.model_param
        gp_model.update_model(True)
        return gp_model
        
    def predict(self, gp_model, theta: np.array):
        """
        predict: preidct slip at theta using given (trained) gp model

        :param gp_model: gaussian process model
        :param theta: terrain inclination
        """
        s_mean, s_var = gp_model.predict(theta)
        return s_mean, s_var

class SlipModelsGenerator:

    def __init__(self, dirname: str, n_terrains: int = 10, is_update: bool = False, type_noise: str = "uniform"):
        """
        __init__: 

        :param dirname: name of directory
        :param n_terrains: number of distincitive terrain classes
        :param is_update: update slip models or not
        :param type_noise: types of noises ("uniform" or "random" noises)
        """
        self.dirname = dirname
        self.n_terrains = n_terrains
        self.is_update = is_update
        self.type_noise = type_noise
        self.rng = np.random.default_rng(0)

        self.slip_models = []
        self.gp_models = []

        self.load_models()

    def load_models(self):
        """
        load_models: load slip models 

        """
        # load slip models
        if os.path.isfile(os.path.join(self.dirname, 'slip_models.pkl')) and self.is_update is False:
            with open(os.path.join(self.dirname, 'slip_models.pkl'), mode='rb') as f:
                self.slip_models = pickle.load(f)
            for i in range(self.n_terrains):
                slip_model = self.slip_models[i]
                gp_model = GPy.models.GPRegression(slip_model.x[:, None], slip_model.y[:, None], kernel=GPy.kern.RBF(1), initialize=False)
                gp_model = slip_model.load_gpr(gp_model)
                self.gp_models.append(gp_model)
        else:
            # generate slip models for each terrain
            for i in range(self.n_terrains):
                slip_model = SlipModel(seed=i)
                self.slip_models.append(slip_model)
            # set GP conditions
            x_ = np.concatenate((np.linspace(-20, 20, 60), np.linspace(-30, -20, 15), np.linspace(20, 30, 15), np.linspace(-40, -30, 5), np.linspace(30, 40, 5)))
            if self.type_noise == "diverse":
                self.set_noise_scale()
                x_ = np.concatenate((np.linspace(-20, 20, 20), np.linspace(-30, -20, 5), np.linspace(20, 30, 5)))
            # generate predicted slip models for each terrain using Gaussian process
            for i in range(self.n_terrains):
                slip_model = self.slip_models[i]
                x, y = slip_model.generate_data(x_)
                gp_model = GPy.models.GPRegression(x[:, None], y[:, None], kernel=GPy.kern.RBF(1))
                gp_model = slip_model.fit_gpr(gp_model)
                self.gp_models.append(gp_model)
            with open(os.path.join(self.dirname, 'slip_models.pkl'), mode='wb') as f:
                pickle.dump(self.slip_models, f)

    def set_noise_scale(self):
        """
        set_noise_scale: set noise scale when it isn't uniformly distributed

        """
        gains = np.zeros(self.n_terrains)
        # get gains for latent functions
        for i, slip_model in enumerate(self.slip_models):
            gains[i] = slip_model.gain
        idx_sort = np.argsort(gains)
        scales_l = [0.050, 0.100]
        scales_m = [0.125, 0.150]
        scales_h = [0.175, 0.200]
        for i, slip_model in enumerate(self.slip_models):
            if gains[i] <= gains[idx_sort[int(self.n_terrains / 3 - 1)]]:
                slip_model.scale = self.rng.choice(scales_l)
            elif gains[idx_sort[int(self.n_terrains / 3 - 1)]] < gains[i] <= gains[idx_sort[int(self.n_terrains * 2 / 3 - 1)]]:
                slip_model.scale = self.rng.choice(scales_m)
            else:
                slip_model.scale = self.rng.choice(scales_h)

    def get_actual_slip(self, theta: float, tf: int):
        """
        get_actual_slip: get actual (w/o randomness) slip ratio for given terrain inclination

        :param theta: terrain inclination in pitch direction
        :param tf: terrain feature (class of distinctive terrain)
        """
        s_gt = self.slip_models[tf].latent_model(theta)
        return (s_gt, None)

    def observe_noisy_slip(self, theta: float, tf: int):
        """
        observe_noisy_slip: observe noisy (w/ randomness) slip ratio for given terrain inclination when path execution (after traversal)
        
        :param theta: terrain inclination in pitch direction
        :param tf: terrain feature (class of distinctive terrain)
        """
        s_obs = self.slip_models[tf].observe_slip(theta)
        return (s_obs, None)
        
    def predict_slip(self, theta: float, tf: int):
        """
        predict_slip: predict slip ratio for given terrain inclination when path planning (before traversal)

        :param theta: terrain inclination in pitch direction
        :param tf: terrain feature (class of distinctive terrain)
        """
        s_mean, s_var = self.slip_models[tf].predict(self.gp_models[tf], np.array([[theta]]))
        return (s_mean[0, 0], s_var[0, 0])

    def visualize(self, fig: plt.figure = None, rc_ax1: int = 121, rc_ax2: int = 122):
        """
        visualize: visualize actual and predicted slip models

        :param rc_ax1: position of plot (actual model)
        :param rc_ax2: position of plot (predicted model)
        """

        sns.set()
        if not fig:
            fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(rc_ax1)
        ax2 = fig.add_subplot(rc_ax2)
        color = cm.jet(np.linspace(0, 1, self.n_terrains))
        for i in range(self.n_terrains):
            theta = np.linspace(-40, 40, 100)
            model_actual = np.zeros(theta.shape[0])
            for j, theta_ in enumerate(theta):
                model_actual[j] = self.slip_models[i].latent_model(theta=theta_)
            ax1.plot(theta, model_actual, color=color[i])
            self.gp_models[i].plot_mean(ax=ax2, color=color[i], label=None)
            self.gp_models[i].plot_confidence(ax=ax2, color=color[i], label=None)
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

def main():
    idx_ds = 3
    dirname = os.path.join(BASE_PATH, '../../datasets/data%02d/' % (idx_ds))
    smg = SlipModelsGenerator(dirname, n_terrains=8, is_update=True, type_noise="diverse")
    smg.visualize()
    plt.show()

if __name__ == '__main__':
    main()