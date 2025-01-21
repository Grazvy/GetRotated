import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate


def rotate_images(images, degrees=-20):
    return np.array([rotate(image, degrees, reshape=False, order=2) for image in images])

def plot_image(image):
    plt.imshow(image, cmap="gray")
    plt.show()


def line_image(n, thickness=2):
    line_image = np.zeros((n, n))
    center_col = n // 2

    if thickness % 2 != 0 or center_col % 2 != 0:
        raise ValueError("Image shapes must be odd.")

    for row in range(center_col, -1, -1):
        for i in range(thickness):
            line_image[row, center_col - thickness // 2 + i] = 1

    return line_image
