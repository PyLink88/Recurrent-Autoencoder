
"""
-Create config file
"""

import json

config_rnn_ae = {

    # Experiment information
    "exp_name": "rnn_ae_ECG5000_exp_0",
    "agent": "RecurrentAEAgent",

    # Architecture hyperparameters
    "rnn_type": "GRU",
    "rnn_act": "None",
    "n_layers": 1,
    "latent_dim": 32,
    "n_features": 1,

    # Optimization hyperparameters
    "learning_rate": 0.001,
    "batch_size": 128,
    "batch_size_val": 256,
    "max_epoch": 2000,

    # Loss function
    'loss': 'MAE',

    # Folder where to retrieve the data and their names
    "data_folder": "./data/ECG5000/numpy/",
    "X_train": "X_train.npy",
    "y_train": "y_train.npy",
    "X_val": "X_val.npy",
    "y_val": "y_val.npy",
    "X_test": "X_test.npy",
    "y_test": "y_test.npy",
    "X_val_p": "X_val_p.npy",
    "y_val_p": "y_val_p.npy",

    # Training type: by now set equal to "one_class"
    "training_type": "one_class",
    "validation_type": "one_class",

    # Checkpoints
    "checkpoint_file": "checkpoint.pth.tar",
    "checkpoint_dir": "./experiments/checkpoints/",
    "load_checkpoint": False,

    # GPU settings
    "cuda": False,
    "device": "cpu",
    "gpu_device": 0,
    "seed": 58
}



if __name__ == '__main__':
    myJSON = json.dumps(config_rnn_ae)
    with open("./configs/config_rnn_ae.json", "w") as jsonfile:
        jsonfile.write(myJSON)
        
        print("Config successfully written")


