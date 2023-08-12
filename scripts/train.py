"""
description: training script for multi-class terrain classification 
author: Masafumi Endo
"""

import random
import sys, os
BASE_PATH = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_PATH, '../planning_project'))

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp

from planning_project.models.unet import Unet
from planning_project.utils.data import DataSet, create_int_label
from planning_project.utils.viz import viz_terrain_props
from planning_project.utils.structs import TrainParams

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def fix_randomness(seed):
    """
    fix_randomness: fix randomness for reproductivity

    :param seed: seed
    """
    # python random
    random.seed(seed)
    # numpy 
    np.random.seed(seed)
    # pytorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

class Runner:

    def __init__(self, model, train_params, seed: int = 0):
        """
        __init__:

        :param model: network architecture
        :param dirname: directory name to dataset
        """
        self.seed = seed

        self.model = model
        self.train_params = train_params

        # make directory if not
        if not os.path.exists(self.train_params.nn_model_dir):
            os.makedirs(self.train_params.nn_model_dir, exist_ok=True)
        if not os.path.exists(self.train_params.log_dir):
            os.makedirs(self.train_params.log_dir, exist_ok=True)
        self.loss = smp.utils.losses.CrossEntropyLoss()
        self.metrics = [smp.utils.metrics.Accuracy(self.train_params.th)]
        self.optimizer = torch.optim.Adam([dict(params=self.model.parameters(), lr=self.train_params.lr)])
        
    def run_trainval(self):
        """
        run_trainval: do training and validation for given network model

        """
        train_loader = self.create_dataloader("train", batch_size=self.train_params.batch_size_train, shuffle=True)
        valid_loader = self.create_dataloader("valid", batch_size=self.train_params.batch_size_valid, shuffle=False)
        train_epoch, valid_epoch = self.create_epoch_runner()
        # main loop for network training
        max_score = self.train_params.max_score
        writer = SummaryWriter(log_dir=self.train_params.log_dir)
        for i in range(self.train_params.num_epochs):
            print(f"Epoch:{i+1}")
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)
            # add logs to tensorboard
            writer.add_scalar("train/loss", train_logs["cross_entropy_loss"], i)
            writer.add_scalar("train/accuracy", train_logs["accuracy"], i)
            writer.add_scalar("valid/loss", valid_logs["cross_entropy_loss"], i)
            writer.add_scalar("valid/accuracy", valid_logs["accuracy"], i)
            if max_score < valid_logs["accuracy"]:
                max_score = valid_logs["accuracy"]
                torch.save(self.model, self.train_params.nn_model_dir + "best_model.pth")
                print("Model saved!")
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                print(f"not improve for {early_stop_counter}Epoch")
                if early_stop_counter == self.train_params.patience:
                    print(f"early stop. Max Score {max_score}")
                    break
            if i == self.train_params.epoch_lr:
                self.optimizer.param_groups[0]["lr"] = 1e-5
                print("Decrease decoder learning rate to 1e-5")
        writer.close()

    def predict(self, color_map):
        """
        predict: predict terrain classes using trained networks

        :param color_map: color map as network inputs
        """
        best_model = torch.load(self.train_params.nn_model_dir + 'best_model.pth', map_location=torch.device(DEVICE))
        input = torch.tensor(color_map).unsqueeze(0)
        pred = best_model(input.to(DEVICE)) # torch.size([1, dim, n, n])
        pred = pred[0].cpu().detach().numpy() # np.array([dim, n, n])
        pred = np.argmax(pred, axis=0) # np.array([n, n])
        return pred

    def create_dataloader(self, split: str, batch_size: int, shuffle: bool):
        """
        create_dataloader: create data loader for network input/output

        :param split: data split (train, valid, or test)
        :param batch_size: batch size
        :param shuffle: whether to shuffle samples
        """
        g = torch.Generator()
        g.manual_seed(self.seed)
        dataset = DataSet(self.train_params.data_dir, split)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            worker_init_fn=self.seed_worker, 
            generator=g)
        return dataloader

    def seed_worker(self, worker_id):
        """
        seed_worker: fix randomness for data loader

        :param worker_id:
        """
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def create_epoch_runner(self):
        """
        create_epoch_runner: create epoch runners for training and validation

        """
        train_epoch = smp.utils.train.TrainEpoch(
            self.model,
            loss=self.loss,
            metrics=self.metrics,
            optimizer=self.optimizer,
            device=DEVICE
        )
        valid_epoch = smp.utils.train.ValidEpoch(
            self.model,
            loss=self.loss,
            metrics=self.metrics,
            device=DEVICE
        )
        return train_epoch, valid_epoch

def predict_likelihood(color_map, train_params):
    """
    predict: predict terrain classes using trained networks

    :param color_map: color map as network inputs
    """
    best_model = torch.load(train_params.nn_model_dir + 'best_model.pth', map_location=torch.device(DEVICE))
    input = torch.tensor(color_map).unsqueeze(0)
    pred = best_model(input.to(DEVICE)) # torch.size([1, dim, n, n])
    pred = pred[0].cpu().detach().numpy() # np.array([dim, n, n])
    pred = np.max(pred, axis=0) # np.array([n, n])
    return pred

def prediction(runner, train_params: str, iters: int = 5):
    """
    prediction: pick up several samples and predict corresponding terrain classes

    :param runner: object for prediction
    :param iters: # of iterations
    """
    dataset = DataSet(train_params.data_dir, "test")
    for i in range(iters):
        color_map, mask = dataset[i]
        pred = runner.predict(color_map)
        likelihood = predict_likelihood(color_map, train_params)
        viz_terrain_props(
            vmin=0,
            vmax=9,
            image=dataset.to_image(color_map), 
            mask=create_int_label(dataset.to_image(mask)), 
            prediction=pred,
            likelihood=likelihood)
    plt.show()

def main():
    seed = 0
    fix_randomness(seed=seed)

    n_ds = 'Std'
    if n_ds != 'AA':
        n_terrains = 10
    else:
        n_terrains = 8
    data_dir = os.path.join(BASE_PATH, '../datasets/dataset_%s/' % (n_ds))
    nn_model_dir = os.path.join(BASE_PATH, '../trained_models/models/dataset_%s/' % (n_ds))
    log_dir = os.path.join(BASE_PATH, '../trained_models/logs/dataset_%s/' % (n_ds))
    is_trainval = True

    model = Unet(n_terrains=n_terrains, in_channels=3).set_model()
    train_params = TrainParams(data_dir, nn_model_dir, log_dir)
    runner = Runner(model, train_params, seed=seed)
    if is_trainval:
        runner.run_trainval()
    else:
        prediction(runner, train_params)
        
if __name__ == '__main__':
    main()