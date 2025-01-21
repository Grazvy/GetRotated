import torch
import numpy as np

class DataLoader:
    def __init__(self, image_size, n_samples, batch_size):
        self.image_size = image_size
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.indices = np.arange(self.n_samples)
        self._generate_data()

    def _generate_data(self):
        self.data = np.random.uniform(0, 1, (self.n_samples, self.image_size, self.image_size))
        self.data = torch.tensor(self.data, dtype=torch.float32)

    def __iter__(self):
        self.current_index = 0
        self._generate_data()   # generate new data every epoch
        return self

    def __next__(self):
        if self.current_index >= self.n_samples:
            raise StopIteration

        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        batch_data = self.data[batch_indices]
        self.current_index += self.batch_size

        return batch_data, None

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size