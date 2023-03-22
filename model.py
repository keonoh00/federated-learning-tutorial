import torch
from torch.utils.data import DataLoader
from utils import DeviceDataLoader, get_device
import pdb


class FederatedNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=20, kernel_size=7)
        self.conv2 = torch.nn.Conv2d(in_channels=20, out_channels=40, kernel_size=7)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(in_features=2560, out_features=10)
        self.relu = torch.nn.functional.relu
        self.track_layers = {
            "conv1": self.conv1,
            "conv2": self.conv2,
            "linear": self.linear,
        }
        self.device = get_device()

    def forward(self, x_batch):
        out = self.conv1(x_batch)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out

    def get_layers(self):
        return self.track_layers

    def apply_parameters(self, parameters_dict):
        with torch.no_grad():
            for layer_name in parameters_dict:

                self.track_layers[layer_name].weight.data *= 0
                self.track_layers[layer_name].bias.data *= 0
                self.track_layers[layer_name].weight = torch.nn.Parameter(
                    parameters_dict[layer_name]["weight"]
                )
                self.track_layers[layer_name].bias = torch.nn.Parameter(
                    parameters_dict[layer_name]["bias"]
                )

            # This is the correct way to do it, but not sure if it is same as above
            # for layer_name, layer in self.track_layers.items():
            #     layer.weight = torch.nn.Parameter(parameters_dict[layer_name]["weight"])
            #     layer.bias = torch.nn.Parameter(parameters_dict[layer_name]["bias"])

    def get_parameters(self):
        parameters_dict = {}

        for layer_name in self.track_layers:
            parameters_dict[layer_name] = {
                "weight": self.track_layers[layer_name].weight.data,
                "bias": self.track_layers[layer_name].bias.data,
            }

        return parameters_dict

    def batch_accuracy(self, predicted, labels):
        with torch.no_grad():
            _, predictions = torch.max(predicted, dim=1)

            return torch.tensor(
                torch.sum(predictions == labels).item() / len(predictions)
            )

    def _process_batch(self, batch):
        images, labels = batch
        predicted = self(images)
        loss = torch.nn.functional.cross_entropy(predicted, labels)
        accuracy = self.batch_accuracy(predicted, labels)
        return loss, accuracy

    def fit(self, dataset, epochs, lr, batch_size=128, opt=torch.optim.SGD):
        dataset_loader = DeviceDataLoader(
            DataLoader(dataset, batch_size, shuffle=True), self.device
        )

        optimizer = opt(self.parameters(), lr=lr)

        history = []

        for epoch in range(epochs):
            losses = []
            accuracies = []
            for batch in dataset_loader:
                loss, accuracy = self._process_batch(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss.detach()  # What does detach do?

                losses.append(loss)
                accuracies.append(accuracy)

            avg_loss = torch.stack(losses).mean().item()
            avg_accuracy = torch.stack(accuracies).mean().item()
            history.append({"loss": avg_loss, "accuracy": avg_accuracy})

        return history

    def evaluate(self, dataset, batch_size=128):
        dataset_loader = DeviceDataLoader(DataLoader(dataset, batch_size), self.device)

        losses = []
        accuracies = []

        with torch.no_grad():
            for batch in dataset_loader:
                loss, accuracy = self._process_batch(batch)
                losses.append(loss)
                accuracies.append(accuracy)

            avg_loss = torch.stack(losses).mean().item()
            avg_accuracy = torch.stack(accuracies).mean().item()

        return avg_loss, avg_accuracy
