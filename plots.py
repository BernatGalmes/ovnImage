import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

from bern_img_utils.images import binary2RGB

from skimage.segmentation import find_boundaries


def drawCircle(image, center_x, center_y, radius, color):
    """
    Draw a circle onto the image
    :param image: image
        where to draw
    :param center_x: int
        x position of the circle
    :param center_y: int
        y position of the circle
    :param radius: int
        radius of the circle
    :param color: Tuple
        rgb color to draw the circle perimeter
    :return:
    """
    cv2.circle(image, center=(center_x, center_y), radius=radius, color=color, thickness=4)
    cv2.rectangle(image, pt1=(center_x - 5, center_y - 5), pt2=(center_x + 5, center_y + 5), color=(0, 0, 255),
                  thickness=cv2.FILLED)


def drawCircles(image, circles, mark_circle):
    """
    Draw multiple circles onto the image of red color. And draw the 'mark_circle' whith green color
    :param image: rgb image
    :param circles: Circle[]
                    Circle => Tuple(center_x, center_y, radius)
    :param mark_circle: Circle circle to remark
                        Circle => Tuple(center_x, center_y, radius)
    :return:
    """
    for center_x, center_y, radius in circles:
        drawCircle(image, center_x, center_y, radius, (220, 20, 20))
    drawCircle(image, mark_circle.cx, mark_circle.cy, mark_circle.r, (10, 255, 20))
    return image


def plotimg(image, filename=None, title=''):
    """
    Plot a single image in x window or on a file as a figure.

    :param image:
    :param filename:
    :param title:
    :return:
    """
    plt.title(title)
    plt.imshow(image)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def multiplot(images, filename=None, nrows=2, colorbar=False):
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
    :param colorbar: bool
                        true to plot a colorbar near each image
    :return:
    """

    n_img = len(images)
    rows = nrows

    if (n_img % 2) == 0:
        columns = int(n_img / rows) + 1
    else:
        columns = int((n_img / rows) + 1) + 1

    plt.figure(figsize=(20, 14))

    for i in range(0, n_img):
        img = images[i]

        image = img['img']
        title = img['title']

        plt.subplot(rows, columns, i + 1)
        plt.title(title)
        plt.axis('off')
        plt.imshow(image, cmap='Greys')
        if colorbar:
            plt.colorbar()

    if filename:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plots_correlation_matrix(df, labels=None, absolute=False):
    """
    Get correlation matrix of dataframe columns
    :param df: pandas dataframe
    :param labels:  String[]
                    labels of each dataframe column
    :return: matplotlib Axes
            Axes object with the heatmap.
    """
    corr = df.corr()

    if labels is None:
        labels = corr.columns.values

    if absolute:
        corr = np.absolute(corr)
    corrmap = sns.heatmap(corr,
                          xticklabels=labels,
                          yticklabels=labels,
                          center=0)
    corrmap.set_yticklabels(corrmap.get_yticklabels(), rotation=45, fontsize=8)
    corrmap.set_xticklabels(corrmap.get_xticklabels(), rotation=90, fontsize=8)
    return corrmap


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