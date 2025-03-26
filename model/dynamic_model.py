import torch
import torch.nn as nn
from base import BaseModel

class DynamicNet(BaseModel):
    def __init__(self, input_dim=1, output_dim=1, hidden_layers=2, hidden_units=32,
                 hidden_activation='relu', output_activation='linear'):
        super().__init__()

        # Activation function mapping
        self.activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'linear': nn.Identity()
        }

        if hidden_activation not in self.activations or output_activation not in self.activations:
            raise ValueError(f"Invalid activation function. Choose from: {list(self.activations.keys())}")

        layers = []
        in_features = input_dim

        # Build hidden layers
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(self.activations[hidden_activation])
            in_features = hidden_units

        # Output layer
        layers.append(nn.Linear(in_features, output_dim))
        layers.append(self.activations[output_activation])

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
