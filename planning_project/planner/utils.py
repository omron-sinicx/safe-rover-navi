"""
description: a collection of utility functions specific to planner
author: Masafumi Endo
"""

import numpy as np

class BaseRiskMetrics:

    def __init__(self, map, type_embed: str, alpha: float = None, n_samples: int = 1e+4):
        """
        __init__: 

        :param map: map class
        :param type_embed: type to embed uncertainty into cost
        :param alpha: parameter to embed uncertainty
        :param n_samples: number of samples for monte-carlo simulation
        """
        self.map = map

        self.type_embed = type_embed
        self.alpha = alpha
        self.n_samples = n_samples
        self.rng = None

    def embed_uncertainty(self, s_mv: tuple, theta: float, xy_ids: tuple):
        """
        embed_uncertainty: embed uncertainty into cost

        :param s_mv: slip mean and variance
        :param theta: terrain inclination
        :param xy_ids: x- and y-axis indices
        """
        if self.type_embed == "mean":
            s = self.mean(s_mv)
        elif self.type_embed == "mean_std":
            s = self.mean_std(s_mv, theta)
        elif self.type_embed == "var":
            # for reproductivity
            grid_id = self.map.calc_grid_id_from_xy_id(xy_ids[0], xy_ids[1])
            self.rng = np.random.default_rng(grid_id)
            s = self.value_at_risk(s_mv, theta)
        elif self.type_embed == "cvar":
            # for reproductivity
            grid_id = self.map.calc_grid_id_from_xy_id(xy_ids[0], xy_ids[1])
            self.rng = np.random.default_rng(grid_id)
            s = self.conditional_value_at_risk(s_mv, theta)
        return s

    def mean(self, s_mv: tuple):
        """
        mean: just return mean slip value

        :param s_mv: slip mean and variance
        """
        return s_mv[0]

    def mean_std(self, s_mv: tuple, theta: float):
        """
        mean_std: mean-std

        :param s_mv: slip mean and standard deviations
        :param theta: terrain inclination
        """
        if 0 <= theta:
            s = s_mv[0] + self.alpha * np.sqrt(s_mv[1])
        else:
            s = s_mv[0] - self.alpha * np.sqrt(s_mv[1])
        return s

    def value_at_risk(self, s_mv: np.array, theta: float):
        """
        value_at_risk: calculate value-at-risk

        :param s_mv: slip mean and standard deviations
        :param prob: probability weights
        """
        mcs = self.monte_carlo_sim(s_mv)
        # value at risk calculation
        if 0 <= theta:
            var = np.percentile(mcs, 100 * self.alpha)
        else:
            var = np.percentile(mcs, 100 * (1 - self.alpha))
        return var

    def conditional_value_at_risk(self, s_mv: np.array, theta: float):
        """
        conditional_value_at_risk: calculate conditional value-at-risk

        :param s_mv: slip mean and standard deviations
        :param prob: probability weights
        """
        mcs = self.monte_carlo_sim(s_mv)
        # value at risk calculation
        if 0 <= theta:
            var = np.percentile(mcs, 100 * self.alpha)
            cvar = mcs[mcs >= var].mean()
        else:
            var = np.percentile(mcs, 100 * (1 - self.alpha))
            cvar = mcs[mcs <= var].mean()
        return cvar

    def monte_carlo_sim(self, s_mv: np.array):
        """
        monte_carlo_sim: simulate slip distribution 

        :param s_mv: slip mean and standard deviations
        """
        # monte-carlo simulation
        mcs = self.rng.normal(loc=s_mv[0], scale=np.sqrt(s_mv[1]), size=int(self.n_samples))
        return mcs

class GMMRiskMetrics:

    def __init__(self, smg, map, pred: np.array, type_embed: str, alpha: float = None, n_samples: int = 1e+4):
        """
        __init__: 

        :param smg: slip model generator class containing GP mean and variance
        :param map: map class
        :param pred: probability distribution of semantic segmentation prediction
        :param type_embed: type to embed uncertainty into cost
        :param alpha: parameter to embed uncertainty
        """
        self.smg = smg
        self.map = map
        self.pred = pred
        self.type_embed = type_embed
        self.alpha = alpha
        self.n_samples = n_samples
        self.rng = None

    def embed_uncertainty(self, theta: float, xy_ids: tuple):
        """
        embed_uncertainty: embed uncertainty into cost

        :param theta: terrain inclination
        :param xy_ids: x- and y-axis indices
        """
        s_mv, prob = self.predict_slips(theta, xy_ids)
        if self.type_embed == "mean":
            s = self.mean(s_mv, prob)
        elif self.type_embed == "mean_std":
            s = self.mean_std(s_mv, prob, theta)
        elif self.type_embed == "var":
            # for reproductivity
            grid_id = self.map.calc_grid_id_from_xy_id(xy_ids[0], xy_ids[1])
            self.rng = np.random.default_rng(grid_id)
            s = self.value_at_risk(s_mv, prob, theta)
        elif self.type_embed == "cvar":
            # for reproductivity
            grid_id = self.map.calc_grid_id_from_xy_id(xy_ids[0], xy_ids[1])
            self.rng = np.random.default_rng(grid_id)
            s = self.conditional_value_at_risk(s_mv, prob, theta)
        return s

    def mean(self, s_mv: np.array, prob: np.array):
        """
        mean: just return mean slip value

        :param s_mv: slip mean and variance
        :param prob: probability weights
        """
        s_mean, _ = self.calculate_mv(s_mv, prob)
        return s_mean

    def mean_std(self, s_mv: np.array, prob: np.array, theta: float):
        """
        mean_std: mean-std

        :param s_mv: slip mean and standard deviations
        :param prob: probability weights
        :param theta: terrain inclination
        """
        s_mean, s_var = self.calculate_mv(s_mv, prob)
        if 0 <= theta:
            s = s_mean + self.alpha * np.sqrt(s_var)
        else:
            s = s_mean - self.alpha * np.sqrt(s_var)
        return s

    def value_at_risk(self, s_mv: np.array, prob: np.array, theta: float):
        """
        value_at_risk: calcualte value-at-risk
        
        :param s_mv: slip mean and standard deviations
        :param prob: probability weights
        :param theta: terrain inclination
        """
        mcs = self.monte_carlo_sim(s_mv, prob)
        # value at risk calculation
        if 0 <= theta:
            var = np.percentile(mcs, 100 * self.alpha)
        else:
            var = np.percentile(mcs, 100 * (1 - self.alpha))
        return var

    def conditional_value_at_risk(self, s_mv: np.array, prob: np.array, theta: float):
        """
        conditional_value_at_risk: calculate conditional value-at-risk

        :param s_mv: slip mean and standard deviations
        :param prob: probability weights
        :param theta: terrain inclination
        """
        mcs = self.monte_carlo_sim(s_mv, prob)
        # conditional value at risk calculation
        if 0 <= theta:
            var = np.percentile(mcs, 100 * self.alpha)
            cvar = mcs[mcs >= var].mean()
        else:
            var = np.percentile(mcs, 100 * (1 - self.alpha))
            cvar = mcs[mcs <= var].mean()
        return cvar

    def monte_carlo_sim(self, s_mv: np.array, prob: np.array):
        """
        monte_carlo_sim: simulate slip distribution 

        :param s_mv: slip mean and standard deviations
        :param prob: probability weights
        """
        # monte-carlo simulation
        mcs = np.empty(0)
        for i in range(self.smg.n_terrains):
            # get mean-variance and probability weights 
            s_mean_, s_var_ = s_mv[i, 0], s_mv[i, 1]
            prob_ = prob[i]
            n_samples_ = int(prob_ * self.n_samples)
            mcs_ = self.rng.normal(loc=s_mean_, scale=np.sqrt(s_var_), size=n_samples_)
            mcs = np.concatenate([mcs, mcs_])
        return mcs

    def predict_slips(self, theta: float, xy_ids: tuple):
        """
        predict_slip: predict slip based on GMM approach

        :param theta: terrain inclination
        :param xy_ids: x- and y-axis indices
        """
        s_mv = np.zeros((self.smg.n_terrains, 2))
        prob = np.zeros(self.smg.n_terrains)
        for i in range(self.smg.n_terrains):
            p_tf = self.pred[i, xy_ids[1], xy_ids[0]] # probability weights
            s_mean_, s_var_ = self.smg.predict_slip(theta, i) # ith terrain GP slip model
            prob[i] = p_tf
            s_mv[i, 0], s_mv[i, 1] = s_mean_, s_var_
        return s_mv, prob

    def calculate_mv(self, s_mv: np.array, prob: np.array):
        """
        calculate_mv: calculate mean and variance based on independent prediction results

        :param s_mv: slip mean and variance
        :param prob: probability weights
        """
        s_mean, s_var = 0, 0
        for i in range(self.smg.n_terrains):
            s_mean_, s_var_ = s_mv[i, 0], s_mv[i, 1]
            prob_ = prob[i]
            s_mean += prob_ * s_mean_
            s_var += prob_**2 * s_var_
        return s_mean, s_var