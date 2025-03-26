import torch
from torch.utils.data import Dataset
import numpy as np
from base import BaseDataLoader

class FunctionDataset(Dataset):
    def __init__(self, n_samples=100, function='linear'):
        self.n_samples = n_samples
        self.function = function.lower()
        self.x = np.random.uniform(0, 2 * np.pi, size=(n_samples, 1))
        self.y = self._generate_y(self.x, self.function)

        # Normalize the data (mean 0, std 1)
        self.x = (self.x - np.mean(self.x)) / np.std(self.x)
        self.y = (self.y - np.mean(self.y)) / np.std(self.y)

        # Convert to tensors
        self.x = torch.from_numpy(self.x).float()
        self.y = torch.from_numpy(self.y).float()

    def _generate_y(self, x, function):
        ε = np.random.uniform(-1, 1, size=x.shape)
        if function == 'linear':
            return 1.5 * x + 0.3 + ε
        elif function == 'quadratic':
            return 2 * x**2 + 0.5 * x + 0.3 + ε
        elif function == 'harmonic':
            return 0.5 * x**2 + 5 * np.sin(x) + 3 * np.cos(3 * x) + 2 + ε
        else:
            raise ValueError(f"Unsupported function type: {function}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.x[index], self.y[index]

class FunctionDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, function='linear', n_samples=100):
        self.dataset = FunctionDataset(n_samples=n_samples, function=function)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
