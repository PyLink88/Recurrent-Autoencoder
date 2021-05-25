# Project configuration
config_rnn_ae = {

    # Experiment information
    "agent": "RecurrentAEAgent",

    # Architecture hyperparameters
    "rnn_type": "GRU",
    "rnn_act": "None",
    "n_features": 1,

    # Optimization hyperparameters
    "learning_rate": 0.001,
    "batch_size_val": 256,
    "max_epoch": 3,

    # AUC hyperparameters
    'sampler_random_state': 88,

    # Folder where to retrieve the data and their names
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

    # GPU settings
    "cuda": True,
    "device": "cuda",
    "gpu_device": 0,
    "seed": 58,

    # Tune
    'tune': True
}
