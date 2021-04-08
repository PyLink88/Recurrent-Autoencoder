# RecAE
A PyTorch implementation of [LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection](https://arxiv.org/pdf/1607.00148.pdf)


## Table of Contents:
<!-- Table of contents generated generated by http://tableofcontent.eu -->
- [RecAE-PyTorch](#recae-pytorch)
    - [Project Structure](#project-structure)
    - [Data Preparation](#data-preparation)
    - [Model](#model)
    - [Experiment configs](#experiment-configs)
    - [Usage](#usage)
    - [Requirements](#requirements)
    - [References](#references)
    - [License](#license)


### Project Structure:
The project structure is based on the following [Pytorch Project Template](https://github.com/moemen95/PyTorch-Project-Template)
```
├── agents
|  └── rnn_autoencoder.py # the main training agent for the recurrent NN-based AE
├── graphs
|  └── models
|  |  └── recurrent_autoencoder.py  # recurrent NN-based AE model definition
|  └── losses
|  |  └── MAELoss.py # contains the Mean Absolute Error (MAE) loss
|  |  └── MSELoss.py # contains the Mean Squared Error (MSE) loss
├── datasets  # contains all dataloaders for the project
|  └── ecg5000.py # dataloader for ECG5000 dataset
├── data
|  └── ECG5000  # contains all ECG time series
├── utils # utilities folder containing metrics, checkpoints and arg parsing (configs).
├── main.py

```

### Model
#### Encoder

![alt text](./utils/assets/encoder.png "Loss during training")

![alt text](./utils/assets/decoder.png "Loss during training")
