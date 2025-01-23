import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import rotate


def rotate_flat_image(image, degrees):
    n = int(math.sqrt(len(image)))
    image = image.detach().view(n, n)
    image = rotate(image, degrees, reshape=False, order=2)
    rotated = np.clip(image, 0, 1)

    return rotated


def rotate_images(images, degrees):
    rotated = []

    for image in images:
        image = rotate(image, degrees, reshape=False, order=2)
        image = np.clip(image, 0, 1)
        rotated.append(image)

    return np.array(rotated)


def plot_image(image, reshape=True, colorbar=False):
    if reshape:
        n = int(math.sqrt(image.size(0)))
        image = image.detach().view(n, n)

    plt.imshow(image, cmap="gray")
    if colorbar:
        plt.colorbar()
    plt.show()


def line_image(n, thickness=2):
    line_image = np.zeros((n, n))
    center_col = n // 2

    if thickness % 2 != 0 or n % 2 != 0:
        raise ValueError("Image shapes must be odd.")

    for row in range(center_col, -1, -1):
        for i in range(thickness):
            line_image[row, center_col - thickness // 2 + i] = 1

    return torch.tensor(line_image, dtype=torch.float32).flatten()


def cat_image():
    car = np.array(Image.open('../resources/cat.jpg'))
    car = 1 - (0.2989 * car[:, :, 0] + 0.5870 * car[:, :, 1] + 0.1140 * car[:, :, 2]) / 255
    return torch.tensor(car, dtype=torch.float32).flatten()


def plot_line_rotation(rotations=17):
    # todo
    pass


def plot_cat_rotation(rotations=17):
    # todo
    pass
