"""
The RNN autoencoder agent class
"""

import torch
from torch import nn
from torch.backends import cudnn

import numpy as np
from tqdm import tqdm
import shutil
import os

from agents.base import BaseAgent
from utils.metrics import AverageMeter
from utils.checkpoints import checkpoints_folder
from utils.config import save_config
from datasets.ecg5000 import ECG500DataLoader 
from graphs.models.recurrent_autoencoder import RecurrentAE
from graphs.losses.MAELoss import MAELoss
from graphs.losses.MSELoss import MSELoss

class RecurrentAEAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

         # Create an instance from the Model
        self.model = RecurrentAE(self.config)

        # Create an instance from the data loader
        self.data_loader = ECG500DataLoader(self.config) # CHANGE

         # Create instance from the loss
        self.loss = MAELoss()

        # Create instance from the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.config.learning_rate)

        # Training info
        self.current_epoch = 0

        # Creating folder where to save checkpoints
        self.checkpoints_path = checkpoints_folder(self.config)

        # Initialize my counters
        self.current_epoch = 0
        self.best_valid = 10e+16 # Setting a very large values
        self.train_loss = np.array([], dtype = np.float64)
        self.valid_loss = np.array([], dtype = np.float64)

        # Check is cuda is available or not
        self.is_cuda = torch.cuda.is_available()
        # Construct the flag and make sure that cuda is available
        self.cuda = self.is_cuda & self.config.cuda

        if self.cuda:
            self.device = torch.device("cuda")
            torch.cuda.manual_seed_all(self.config.seed)
            torch.cuda.set_device(self.config.gpu_device)
            print("Operation will be on *****GPU-CUDA***** ")
            #print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.config.seed)
            print("Operation will be on *****CPU***** ")

        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)

        # Loading chekpoint
        self.load_checkpoint(self.config.checkpoint_file)

    def train(self):

        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            perf_train = self.train_one_epoch()
            perf_valid = self.validate_one_epoch()

            # Updating training and validation loss
            self.train_loss = np.append(self.train_loss, perf_train.avg)
            self.valid_loss = np.append(self.valid_loss, perf_valid.avg)

            # Printing current training and validation loss
            print('Training loss at epoch '+ str(self.current_epoch) + ' is ' + str( perf_train.avg))
            print('Validation loss at epoch '+ str(self.current_epoch) + ' is ' + str( perf_valid.avg))
            
            # Saving
            is_best = perf_valid.sum < self.best_valid
            if is_best:
                self.best_valid = perf_valid.sum
            self.save_checkpoint(is_best=is_best)

    def train_one_epoch(self):
        """ One epoch training step """

        # Initialize tqdm
        tqdm_batch = tqdm(self.data_loader.train_loader, total = self.data_loader.train_iterations,
                         desc ="Epoch-{}-".format(self.current_epoch))

        # Set the model to be in training mode
        self.model.train()

        # Initialize your average meters
        epoch_loss = AverageMeter()

        # One epoch of training
        for x, y in tqdm_batch: 
            if self.cuda:
                x, y = x.cuda(), y.cuda()
                
            # Model
            y_hat = self.model(x)

            # Current training loss
            cur_tr_loss = self.loss(x, y_hat)
           
            if np.isnan(float(cur_tr_loss.item())):
                raise ValueError('Loss is nan during training...')

            # Optimizer
            self.optimizer.zero_grad()
            cur_tr_loss.backward()
            self.optimizer.step()

            # Updating loss
            epoch_loss.update(cur_tr_loss.item())

        tqdm_batch.close()
      
        return epoch_loss

    def validate_one_epoch(self):
        """ One epoch validation step """
        # Initialize tqdm
        tqdm_batch = tqdm(self.data_loader.valid_loader, total = self.data_loader.valid_iterations,
                         desc = "Validation at epoch -{}-".format(self.current_epoch))

        # Set the model to be in evaluation mode
        self.model.eval()

        # Initialize your average meters
        epoch_loss = AverageMeter()

        with torch.no_grad():

            for x, y in tqdm_batch:
                if self.cuda:
                    x, y = x.cuda(), y.cuda()               

                # Model
                y_hat = self.model(x)

                # Current training loss
                cur_val_loss = self.loss(x, y_hat)
                if np.isnan(float(cur_val_loss.item())):
                    raise ValueError('Loss is nan during validation...')

                # Updating loss
                epoch_loss.update(cur_val_loss.item())

            tqdm_batch.close()
        return epoch_loss
    
    def save_checkpoint(self, filename ='checkpoint.pth.tar', is_best = 0):
        """
        Saving the latest checkpoint of the training
        :param filename: filename which will contain the state
        :param is_best: flag is it is the best model
        :return:
        """
        state = {
            'epoch': self.current_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'valid_loss': self.valid_loss,
            'train_loss': self.train_loss
        }

        # Save the state
        torch.save(state, self.checkpoints_path + filename)

        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.checkpoints_path + filename,
                            self.checkpoints_path + 'model_best.pth.tar')
            print('Saving a best model')

    def load_checkpoint(self, filename):

        if self.config.load_checkpoint:
            filename = self.checkpoints_path + filename
            try:
                checkpoint = torch.load(filename)
                self.current_epoch = checkpoint['epoch']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.valid_loss = checkpoint['valid_loss']
                self.train_loss = checkpoint['train_loss']

                print("Checkpoint loaded successfully from '{}' at (epoch {}) \n"
                                .format(self.checkpoints_path , checkpoint['epoch']))
            except OSError as e:
                print("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
        else:
            print('Training a new model from scratch')
            
    def run(self):
        """
        The main operator
        :return:
        """
        # Saving config
        save_config(self.config, self.checkpoints_path)

        # Model training
        self.train()
 
    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        self.save_checkpoint()
        self.data_loader.finalize()