"""
ECG5000 Dataloader implementation, used in RNN_Autoencoder
"""

import numpy as np
from utils.samplers import StratifiedSampler

import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset



class ECG500DataLoader:
    def __init__(self, config):
        self.config = config

        # Loading training data
        if self.config.training_type == 'one_class':
            # If loss without AUC penalty is used
            X_train = np.load(self.config.data_folder + self.config.X_train).astype(np.float32)
            y_train = np.load(self.config.data_folder + self.config.y_train).astype(np.float32)
        else:
            # If loss with AUC penalty is used
            X_train = np.load(self.config.data_folder + self.config.X_train_p).astype(np.float32)
            y_train = np.load(self.config.data_folder + self.config.y_train_p).astype(np.float32)
        
        # Loading validation data to control model training
        X_val = np.load(self.config.data_folder + self.config.X_val).astype(np.float32)
        y_val = np.load(self.config.data_folder + self.config.y_val).astype(np.float32)

        # From numpy to torch
        if X_train.ndim < 3:
            X_train = torch.from_numpy(X_train).unsqueeze(2)
            X_val = torch.from_numpy(X_val).unsqueeze(2)
        else:
            X_train = torch.from_numpy(X_train)
            X_val = torch.from_numpy(X_val)

        y_train = torch.from_numpy(y_train)
        y_val = torch.from_numpy(y_val)

        # Tensordataset
        training = TensorDataset(X_train, y_train)
        validation = TensorDataset(X_val, y_val)

        # Dataloader
        if self.config.training_type == 'one_class':

            self.train_loader = DataLoader(training, batch_size = self.config.batch_size, shuffle = True)
        else:

            sampler = StratifiedSampler(y_train,
                                        batch_size =self.config.batch_size,
                                        random_state =self.config.sampler_random_state)
            self.train_loader = DataLoader(training, batch_sampler = sampler)

        self.valid_loader = DataLoader(validation, batch_size = self.config.batch_size_val, shuffle = False)

        # Number of batches
        self.train_iterations = len(self.train_loader)
        self.valid_iterations = len(self.valid_loader)

    def finalize(self):
        pass



