import cv2
import numpy as np


def binary2RGB(image: np.ndarray) -> np.ndarray:
    """
    Convert a binary image to RGB, black & white image
    O values to BLACK others to white
    :param image: binary image
    :return: binary image coverted to RGB, true values (or 1) to BLACK and other to WHITE
    """
    img = image.astype(np.uint8)
    mask = np.zeros(img.shape, dtype=np.uint8)

    res = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    res[np.equal(img, mask)] = (255, 255, 255)
    return res


def reduce_image(img: np.ndarray, size: int, back_value: int=255) -> np.ndarray:
    """
    Resize the image without deformation
    :param back_value: int value to fill background if not fit perfectly
    :param img: np.ndarray
    :param size: Tuple (witdh, height)
    :return:
    """

    witdh, height = size
    if len(img.shape) > 2:
        shape = (witdh, height, img.shape[2])
    else:
        shape = (witdh, height)
    back = np.full(shape, back_value)
    shap = img.shape

    if shap[0] > shap[1]:
        r_height = witdh
        r_width = int((witdh / shap[0]) * shap[1])
    else:
        r_height = int((height / shap[1]) * shap[0])
        r_width = height

    aux = cv2.resize(img, (r_width, r_height))

    w_off = int((back.shape[0] - aux.shape[0]) / 2)
    h_off = int((back.shape[1] - aux.shape[1]) / 2)

    if len(img.shape) > 2:
        back[w_off:aux.shape[0] + w_off, h_off:aux.shape[1] + h_off, :] = aux
    else:
        back[w_off:aux.shape[0] + w_off, h_off:aux.shape[1] + h_off] = aux

    return back
