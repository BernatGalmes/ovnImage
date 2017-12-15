import numpy as np


def binary2RGB(image):
    """
    Convert a binary image to RGB, black & white image
    O values to BLACK others to white
    :param image: binary image
    :return: binary image coverted to RGB, true values (or 1) to BLACK and other to WHITE
    """
    img = image.astype('uint8')
    mask = np.zeros(img.shape, dtype=np.uint8)

    res = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    res[np.equal(img, mask)] = (255, 255, 255)
    return res


def grays2binary(image_grays):
    """
    Convert a grayscale image to a binary one
    :param image_grays: one chanel image
    :return:
    """
    results = np.zeros(image_grays.shape)
    results[np.not_equal(image_grays, results)] = 1
    return results
