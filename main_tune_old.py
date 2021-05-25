import numpy as np
import pandas as pd
import os
from ray import tune
from easydict import EasyDict
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim

from agents.rnn_autoencoder import RecurrentAEAgent
from graphs.models.recurrent_autoencoder import RecurrentAE
from datasets.common_loader import RecAEDataLoader 

import warnings
warnings.filterwarnings("ignore")

# Project configuration
config_rnn_ae = {

    # Experiment information
    "exp_name": "rnn_ae_ECG5000_exp_0_b",
    "agent": "RecurrentAEAgent",

    # Architecture hyperparameters
    "rnn_type": "GRU",
    "rnn_act": "None",
    "n_layers": 1,
    "n_features": 1,

    # Optimization hyperparameters
    "learning_rate": 0.001,
    "batch_size": 128,
    "batch_size_val": 256,
    "max_epoch": 3,

    # Loss function
    'loss': 'MAEAUC',

    # AUC hyperparameters
    'lambda_auc': 0.1,
    'sampler_random_state': 88,

    # Folder where to retrieve the data and their names (IMPORTANT: it must be a global dir)
    "data_folder":  "/content/drive/MyDrive/Recurrent-Autoencoder-hyper_tuning_NEW/Recurrent-Autoencoder-hyper_tuning_NEW/data/ECG5000/numpy/",
    "X_train": "X_train.npy",
    "y_train": "y_train.npy",
    "X_train_p": "X_train_p.npy",
    "y_train_p": "y_train_p.npy",
    "X_val": "X_val.npy",
    "y_val": "y_val.npy",
    "X_test": "X_test.npy",
    "y_test": "y_test.npy",
    "X_val_p": "X_val_p.npy",
    "y_val_p": "y_val_p.npy",

    # Training type: by now set equal to "one_class"
    "training_type": "more_class",
    "validation_type": "one_class",

    # Checkpoints
    "checkpoint_file": "checkpoint.pth.tar",
    "checkpoint_dir": "./experiments/checkpoints/",
    "load_checkpoint": False,

    # GPU settings
    "cuda": True,
    "device": "cuda",
    "gpu_device": 0,
    "seed": 58,

    # Tune
    'tune': True
}

# From dict to easydict
config_rnn_ae = EasyDict(config_rnn_ae)

# First two parameters must be "config" and "checkpoint_dir"
def tune_model(config, checkpoint_dir = None, config_rnn_ae = None):
    
    #os.chdir('/content/drive/MyDrive/Recurrent-Autoencoder-auc_loss/Recurrent-Autoencoder-auc_loss')

    loss_type, lambda_reg = config['loss_param']
    config['loss_type'] = loss_type
    config['lambda_reg'] = lambda_reg

    # TO DO add the following to directly to the agent class 
    if config["loss_type"] == 'MAE':
      config_rnn_ae.training_type = 'one_class'
    else:
      config_rnn_ae.training_type = 'more_class'

    # Create an instance of the agent
    agent = RecurrentAEAgent(config_rnn_ae)

    # Create an instance from the data loader
    agent.data_loader = RecAEDataLoader(config["batch_size"], agent.config) 

    # Setting the model
    agent.model = RecurrentAE(config["latent_dim"], agent.config)
    agent.model.to(agent.device)

    # Setting the loss
    agent.loss = agent.possible_loss[config["loss_type"]]
    agent.loss.to(agent.device)

    # Setting the optimizer
    agent.optimizer = torch.optim.Adam(agent.model.parameters(), lr = config["lr"])
    agent.train_tune(config['lambda_reg'], checkpoint_dir)

    # Finalizing
    agent.finalize_tune(checkpoint_dir)   
    perf = agent.best_valid

    # Metric to be reported by tune
    tune.report(mean_accuracy = perf)

if __name__ == "__main__":
    
    # Folder where to save experiments the results
    my_dir = "/content/drive/MyDrive/Recurrent-Autoencoder-hyper_tuning_NEW/Recurrent-Autoencoder-hyper_tuning_NEW/experiments"
   
    # Project name
    project_name ='ECG_5000' # Give a name like the dataset

    # Creating nested conditional grid 
    def _iter_loss():
        for loss in ['MAE','MAEAUC']:
            if loss == 'MAE':
                yield loss, 0
            else:
                for lambda_reg in [0.001,0.01, 0.1, 1, 10]:
                    yield loss, lambda_reg

    analysis = tune.run(partial(tune_model, config_rnn_ae = config_rnn_ae), 
                    config = {"latent_dim": tune.grid_search([35, 70, 105]),
                              "lr": tune.grid_search([0.001, 0.01]),
                              "batch_size": tune.grid_search([256]),
                              "loss_param": tune.grid_search(list(_iter_loss()))}, 
                    resources_per_trial = {"cpu": 2, "gpu": 1}, 
                    name = project_name, 
                    local_dir = my_dir)


    # Saving all results
    df = analysis.dataframe()
    df.to_pickle(my_dir + '/df_analysis.pkl')



                    


