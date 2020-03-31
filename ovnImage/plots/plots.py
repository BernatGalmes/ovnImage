"""
Useful functions to easily create new plots
"""
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from ..images import binary2RGB

from skimage.segmentation import find_boundaries
from mpl_toolkits.axes_grid1 import make_axes_locatable


def multiplot(images, filename=None, nrows=2, cmap='Greys'):
    """
    Help to plot a multiple image figure
    :param images: list of dictionaries with structure:
        {
        "img": image to plot,
        "title": name of the plot
        }
    :param filename: String|None
                        Path file Where save the figure,
                        None to plot in a window
    :return:
    """
    plt.clf()
    n_img = len(images)

    if (n_img % 2) == 0:
        columns = int(n_img / nrows)
    else:
        columns = int((n_img / nrows) + 1)

    for i, img in enumerate(images):
        image = img['img']
        title = img['title']

        ax = plt.subplot(nrows, columns, i + 1)
        ax.set_title(title)
        ax.axis('off')
        if 'cmap' in img:
            imshow = ax.imshow(image, cmap=img['cmap'])

        else:
            imshow = ax.imshow(image, cmap=cmap)

        if 'colorbar' in img:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(imshow, cax=cax, orientation='vertical')

    if filename:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()


def plots_segmentation(img, labels):
    """
    Plot the results of an image segmentation
    Example of use:
    >>> from skimage.segmentation import slic
    >>> img = cv2.imread("path/to/image")
    >>> img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    >>> segments = slic(img, n_segments=20, compactness=10, max_iter=100, sigma=2.3)
    >>> plots_segmentation(img, segments)
    :param img: RGB image
    :param labels: 2D or 3D array
                    Integer mask indicating segment labels.
    :return:
    """
    img_res = img.copy()
    boundaries = find_boundaries(labels).astype(np.uint)
    img_res = cv2.addWeighted(img_res, 0.7, binary2RGB(boundaries), 0.3, 0)
    plt.imshow(img_res)
    plt.show()


def plots_raw_data(df, columns, colors="b"):
    """
    Plot raw data of the given dataframe
    :param df: DataFrame Wich contains all the data
    :param columns: List strings name of the columns to plot
    :param colors: color, sequence, or sequence of color, optional, default: ‘b’
                    c can be a single color format string, or a sequence of color specifications of length N,
                    or a sequence of N numbers to be mapped to colors using the cmap and norm specified
                    via kwargs (see below).
                    Note that c should not be a single numeric RGB or RGBA sequence because that is indistinguishable
                    from an array of values to be colormapped. c can be a 2-D array in which the rows are RGB or RGBA,
                    however, including the case of a single row to specify the same color for all points.
    :return:
    """
    if len(columns) == 3:
        projection = "3d"
    else:
        projection = "rectilinear"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection=projection)
    if projection == "3d":
        ax.scatter(df[columns[0]], df[columns[1]], df[columns[2]], c=colors, marker='o')
        ax.set_xlabel(columns[0])
        ax.set_ylabel(columns[1])
        ax.set_zlabel(columns[2])
    else:
        ax.scatter(df[columns[0]], df[columns[1]], c=colors, marker='o')
        ax.set_xlabel(columns[0])
        ax.set_ylabel(columns[1])
    plt.show()


def draw_mask_boundaries(img: np.ndarray, mask: np.ndarray, color: tuple, thickness: int = 2):
    """
    Draw the boundaries of a a given mask onto an image.

    :param img: RGB image to draw onto.
    :param mask: Binary image to use as mask.
    :param color: tuple of the rgb color to use to draw the contour.
    :param thickness: Thickness of lines the contours are drawn with.
    :return:
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(img, contours, contourIdx=-1, thickness=thickness, color=color,
                     lineType=cv2.LINE_8)
