import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
