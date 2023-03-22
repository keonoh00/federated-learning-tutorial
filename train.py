# Source: https://www.kaggle.com/code/puru98/federated-learning-pytorch

import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from client import Client
from model import FederatedNet
from utils import get_device, to_device
import pdb

# ?????
# %matplotlib inline
# plt.rcParams["figure.figsize"]=[5,5]


if __name__ == "__main__":
    train_dataset = MNIST(
        "./data", train=True, download=True, transform=transforms.ToTensor()
    )
    test_dataset = MNIST(
        "./data", train=False, download=True, transform=transforms.ToTensor()
    )

    # Split the dataset into 100 parts
    train_dataset, dev_dataset = random_split(
        train_dataset, [int(len(train_dataset) * 0.83), int(len(train_dataset) * 0.17)]
    )

    # Hyperparameters
    classes = 10
    input_dim = 784
    num_clients = 100
    rounds = 30
    batch_size = 128
    epochs_per_client = 3
    learning_rate = 2e-2

    total_train_size = len(train_dataset)
    total_test_size = len(test_dataset)
    total_dev_size = len(dev_dataset)

    # Set up clients
    examples_per_client = total_train_size // num_clients
    client_datasets = random_split(
        train_dataset,
        [
            min(i + examples_per_client, total_train_size) - i
            for i in range(0, total_train_size, examples_per_client)
        ],
    )

    clients = [Client(i, client_datasets[i]) for i in range(num_clients)]

    # Set up server
    global_model = to_device(FederatedNet(), get_device())
    history = []

    # Train
    for curr_round in range(rounds):
        print(f"Round {curr_round + 1} / {rounds}")

        curr_parameters = global_model.get_parameters()
        new_parameters = dict(
            [(layer_name, {"weight": 0, "bias": 0}) for layer_name in curr_parameters]
        )

        for client in clients:
            client_parameters = client.train(
                curr_parameters, epochs_per_client, learning_rate, batch_size
            )
            fraction = client.get_dataset_size() / total_train_size
            for layer_name in client_parameters:
                new_parameters[layer_name]["weight"] += (
                    fraction * client_parameters[layer_name]["weight"]
                )
                new_parameters[layer_name]["bias"] += (
                    fraction * client_parameters[layer_name]["bias"]
                )
        global_model.apply_parameters(new_parameters)

        train_loss, train_acc = global_model.evaluate(train_dataset)
        dev_loss, dev_acc = global_model.evaluate(dev_dataset)

        print(
            f"Train Loss: {round(train_loss,4)} - Train Accuracy: {round(train_acc,4)}"
        )
        print(f"Dev Loss: {round(dev_loss,4)} - Dev Accuracy: {round(dev_acc,4)}")

        history.append((train_loss, dev_loss))
    # TODO: Save the histroy
