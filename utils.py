import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate


def rotate_images(images, degrees=-20):
    rotated = []

    for image in images:
        image = rotate(image, degrees, reshape=False, order=2)
        image = np.clip(image, 0, 1)
        rotated.append(image)

    return torch.tensor(np.array(rotated), dtype=torch.float32)


def plot_image(image, colorbar=False):
    plt.imshow(image.detach(), cmap="gray")
    if colorbar:
        plt.colorbar()
    plt.show()


def line_image(n, thickness=2):
    line_image = np.zeros((n, n))
    center_col = n // 2

    if thickness % 2 != 0 or center_col % 2 != 0:
        raise ValueError("Image shapes must be odd.")

    for row in range(center_col, -1, -1):
        for i in range(thickness):
            line_image[row, center_col - thickness // 2 + i] = 1

    return torch.tensor(line_image, dtype=torch.float32)
