from model import FederatedNet
from utils import to_device, get_device


class Client:
    def __init__(self, client_id, dataset):
        self.client_id = client_id
        self.dataset = dataset
        self.model = FederatedNet()
        self.device = get_device()

    def get_dataset_size(self):
        return len(self.dataset)

    def get_client_id(self):
        return self.client_id

    def train(self, parameters_dict, epochs=1, lr=0.01, batch_size=128):
        model = to_device(self.model, self.device)
        model.apply_parameters(parameters_dict)

        train_history = model.fit(self.dataset, epochs, lr, batch_size)

        print(
            f"Client {self.client_id} - Training Loss: {round(train_history[-1]['loss'],4)} - Training Accuracy: {round(train_history[-1]['accuracy'],4)}"
        )

        return model.get_parameters()
