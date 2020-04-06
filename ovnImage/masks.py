import cv2
import numpy as np

import sklearn.metrics as metrics


def bounding_box(mask: np.ndarray):
    """
    Get the minimum enclosing rectangle of a given mask.

    :param mask: binary image
    :return: None if it is an empty mask or th x, y, width, height bounding box values
    """
    indexs = mask.nonzero()
    if len(indexs[0]) == 0:
        return None

    # Get the bounding box
    y_min, y_max = indexs[0].min(), indexs[0].max()
    x_min, x_max = indexs[1].min(), indexs[1].max()

    # ignore too small bounding boxes
    if y_min == y_max or x_min == x_max:
        return None
    return x_min, y_min, x_max - x_min, y_max - y_min


def bounding_circle(mask: np.ndarray):
    """
    Get the minium enclosing circle of a given mask
    :param mask:
    :return: (x, y), r => integers
    """
    _, th = cv2.threshold(mask.copy(), 1, 255, cv2.THRESH_BINARY_INV)
    _, contours, _ = cv2.findContours(th.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]

    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    return center, radius


def evaluation(mask: np.ndarray, ground_truth: np.ndarray):
    """
    Get the evaluation metrics of a given mask and his likelihood
    :param mask: Binary image => Mask to evaluate
    :param ground_truth: Binary image => Correct mask
    :raise Incorrect likelihood
    :return: dict{
        "FN"
        "TP"
        "FP"
        "Recall"
        "Precision"
        "F1"
        "cohen_kappa"
        "accuracy"
    }

    """
    stats = {}

    likelihood = np.copy(ground_truth).astype(np.uint8)
    P = np.count_nonzero(ground_truth)  # the number of real positive cases in the data
    invert_likelihood = np.bitwise_not(likelihood)
    N = np.count_nonzero(invert_likelihood)  # the number of real negative cases in the data

    if N == 0 or P == 0:
        raise Exception("Incorrect likelihood")

    # pixels identificats com a negatius que haurien de ser positius
    fn = np.count_nonzero(np.bitwise_and(np.bitwise_not(mask), likelihood))

    # pixels identificats com a positius que haurien de ser positius
    tp = np.count_nonzero(np.bitwise_and(mask, likelihood))

    # pixels identificats com a positius que haurien de ser negatius
    fp = np.count_nonzero(np.bitwise_and(mask, np.bitwise_not(likelihood)))

    tn = N - fp

    stats["FNR"] = fn / P
    stats["FPR"] = fp / N

    stats["TPR"] = tp / P
    stats["TNR"] = tn / N

    if stats["TPR"] != 0 or stats["FPR"] != 0:
        stats["Precision"] = tp / (tp + fp)
    else:
        stats["Precision"] = 0

    if stats["Precision"] != 0 or stats["TPR"] != 0:
        stats["F1"] = (2 * stats["Precision"] * stats["TPR"]) / (stats["Precision"] + stats["TPR"])
        stats["accuracy"] = (tp + tn) / (P + N)
        stats["cohen_kappa"] = metrics.cohen_kappa_score(likelihood.flatten(), mask.flatten())

    return stats


def sklearn_evaluation(mask: np.ndarray, ground_truth: np.ndarray, pos_label=255):
    """
    Same than mask_evaluation, but using sklearn library for compute values
    :param mask:
    :param ground_truth:
    :param pos_label: value of the true values
    :return:
    """
    y_true = ground_truth.flatten()
    y_pred = mask.flatten()

    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()

    N = tn + fp
    P = fn + tp

    prfs = metrics.precision_recall_fscore_support(y_true, y_pred, pos_label=pos_label, average="binary")

    stats = {
        "Precision": tp / (tp + fp),
        "TPR": prfs[1],
        "TNR": tn / N,
        "FPR": fp / N,
        "FNR": fn / P,
        "Fbeta": prfs[2],
        "cohen_kappa": metrics.cohen_kappa_score(y_true, y_pred),
        "accuracy": (tp + tn) / (P + N),
        "r2": metrics.r2_score(y_true, y_pred),
        "F1": metrics.f1_score(y_true, y_pred, pos_label=pos_label)
    }

    return stats


def coincidence(mask1, mask2, priority="big_mask"):
    """
    Get the percentage of coincident pixels between two masks of the same shape.

    If priority is 'small_mask' then the percentage is over the number of pixels of the smaller mask.

    :param priority:
    :param mask1:
    :param mask2:
    :return:
    """
    mask1 = mask1.astype(np.uint8)
    mask2 = mask2.astype(np.uint8)

    res = np.bitwise_and(mask1, mask2)

    # number of coincident pixels
    equals = np.count_nonzero(res)
    n_pix1 = np.count_nonzero(mask1)
    n_pix2 = np.count_nonzero(mask2)

    if priority == "small_mask":
        max_pix = np.min([n_pix1, n_pix2])
    else:
        max_pix = np.max([n_pix1, n_pix2])

    return equals / max_pix


def onto_mask(mask1, mask2, perc=0.9):
    """
    Given two masks of the same shape, check if them are one onto the other.

    :param mask1:
    :param mask2:
    :param perc: percentage of coincidence pixels that must have the two masks
                    to consider that them are one onto the other.
    :return: bool
    """
    res = np.bitwise_and(mask1.astype(np.bool), mask2.astype(np.bool))

    # number of coincident pixels
    equals = np.count_nonzero(res)

    n_pix1 = np.count_nonzero(mask1)
    n_pix2 = np.count_nonzero(mask2)

    return equals > (n_pix1 * perc) or equals > (n_pix2 * perc)


def fill_holes(mask):
    """
    Fill all the empty pixels overwhelmed by true pixels.

    :param mask: 0, 255 mask
    :return: O values inside 255 values are filled with 255
    """
    # Threshold.
    # Set values equal to or above 220 to 0.
    # Set values below 220 to 255.
    mask = mask.copy()
    # mask = cv2.bitwise_not(mask)
    if mask.max() == 1:
        im_th = mask * 255

    else:
        mask = cv2.bitwise_not(mask)
        th, im_th = cv2.threshold(mask, 220, 255, cv2.THRESH_BINARY_INV)

    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    if mask.max() == 1:
        im_out = im_out / 255

    return im_out.astype(np.uint8)


def delete_contour_in(mask, region):  # TODO: comment function
    """

    :param mask:
    :param region:
    :return:
    """
    mask = mask.astype(np.uint8)

    _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:

        area = cv2.contourArea(contour)
        if area > 100:
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            if cy < region:
                cv2.drawContours(mask, [contour], 0, 0, -1)

    return mask


def biggest_connected_component(mask):
    """
    Get the biggest connected component of a mask.

    :param mask:
    :return:
    """
    mask = mask.copy().astype(np.uint8)
    mask[mask != 0] = np.iinfo(np.uint8).max

    # Find the largest contour and extract it
    im, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return mask

    biggest_cnt = max(contours, key=cv2.contourArea)

    # Create a mask from the largest contour
    mask = np.zeros_like(mask)
    cv2.fillPoly(mask, [biggest_cnt], 1)

    return mask


def every_separated(masks):
    """
    Get an image of every component in the mask with different color

    :param masks:
    :return:
    """
    masks = masks.copy().astype(np.uint8)
    masks[masks != 0] = np.iinfo(np.uint8).max
    # Find the largest contour and extract it
    _, contours, _ = cv2.findContours(masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    multi_mask = np.zeros_like(masks)
    color = 1
    for cnt in contours:
        cv2.fillPoly(multi_mask, [cnt], color)
        color = color + 1

    return multi_mask
