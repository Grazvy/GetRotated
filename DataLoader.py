import torch
import numpy as np
from utils import rotate_images

class DataLoader:
    def __init__(self, image_size, n_samples, batch_size, degrees=-20):
        self.image_size = image_size
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.degrees = degrees
        self.indices = np.arange(self.n_samples)
        self._generate_data()

    def _generate_data(self):
        # todo include multiple rotations
        x = np.random.uniform(0, 1, (self.n_samples, self.image_size, self.image_size))
        y = rotate_images(x, self.degrees)
        self.X = torch.tensor(x.reshape(self.n_samples, -1), dtype=torch.float32)
        self.Y = torch.tensor(y.reshape(self.n_samples, -1), dtype=torch.float32)

    def __iter__(self):
        self.current_index = 0
        self._generate_data()   # generate new data every epoch
        return self

    def __next__(self):
        if self.current_index >= self.n_samples:
            raise StopIteration

        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        batch_x = self.X[batch_indices]
        batch_y = self.Y[batch_indices]
        self.current_index += self.batch_size

        return batch_x, batch_y

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size