import numpy as np
class DataLoader:
    def __init__(self, image_size, n_samples, batch_size):
        # todo remove batch size?
        self.image_size = image_size
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.indices = np.arange(self.n_samples)
        self._generate_data()

    def _generate_data(self):
        #todo
        self.data = None

    def __iter__(self):
        self.current_index = 0
        np.random.shuffle(self.indices)  # Shuffle indices at the start of each epoch
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