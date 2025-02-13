import torch
import numpy as np
from modules.utils import rotate_images


class DataLoader:
    def __init__(self, image_size, n_samples, batch_size, degrees):
        self.image_size = image_size
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.degrees = degrees
        self.indices = np.arange(self.n_samples)
        self.X = None
        self.Y = None


    def _generate_data(self):
        raise NotImplementedError

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


class UniformNoiseLoader(DataLoader):
    def __init__(self, image_size, n_samples, batch_size, degrees=-20):
        super(UniformNoiseLoader, self).__init__(image_size, n_samples, batch_size, degrees)

    def _generate_data(self):
        x = np.random.uniform(0, 1, (self.n_samples, self.image_size, self.image_size))
        y = rotate_images(x, self.degrees)
        self.X = torch.tensor(x.reshape(self.n_samples, -1), dtype=torch.float32)
        self.Y = torch.tensor(y.reshape(self.n_samples, -1), dtype=torch.float32)


class FixedPointsLoader(DataLoader):
    def __init__(self, image_size, n_samples, batch_size, n_points=10, min_value=0, degrees=-20):
        super(FixedPointsLoader, self).__init__(image_size, n_samples, batch_size, degrees)
        self.n_points = n_points
        self.min_value = min_value

    def _generate_data(self):
        x = np.zeros((self.n_samples, self.image_size, self.image_size))

        for i in range(self.n_samples):
            x_coords, y_coords = np.random.randint(0, self.image_size, size=(2, self.n_points))
            values = np.random.uniform(self.min_value, 1, size=self.n_points)
            x[i, x_coords, y_coords] = values

        y = rotate_images(x, self.degrees)
        self.X = torch.tensor(x.reshape(self.n_samples, -1), dtype=torch.float32)
        self.Y = torch.tensor(y.reshape(self.n_samples, -1), dtype=torch.float32)


class RandomPointsLoader(DataLoader):
    def __init__(self, image_size, n_samples, batch_size, max_points=None, degrees=-20):
        super(RandomPointsLoader, self).__init__(image_size, n_samples, batch_size, degrees)
        self.max_points = n_samples if max_points is None else max_points

    def _generate_data(self):
        x = np.zeros((self.n_samples, self.image_size, self.image_size))

        for i in range(self.n_samples):
            size = np.random.randint(0, self.max_points)

            x_coords, y_coords = np.random.randint(0, self.image_size, size=(2, size))
            values = np.random.uniform(0, 1, size=size)
            x[i, x_coords, y_coords] = values

        y = rotate_images(x, self.degrees)
        self.X = torch.tensor(x.reshape(self.n_samples, -1), dtype=torch.float32)
        self.Y = torch.tensor(y.reshape(self.n_samples, -1), dtype=torch.float32)