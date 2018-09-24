import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

from .images import binary2RGB

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
    rows = nrows

    if (n_img % 2) == 0:
        columns = int(n_img / rows)
    else:
        columns = int((n_img / rows) + 1)

    for i in range(0, n_img):
        img = images[i]

        image = img['img']
        title = img['title']

        ax = plt.subplot(rows, columns, i + 1)
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

def plots_correlation_matrix(df, labels=None, absolute=False, method='pearson'):
    """
    Get correlation matrix of dataframe columns
    :param df: pandas dataframe
    :param labels:  String[]
                    labels of each dataframe column
    :return: matplotlib Axes
            Axes object with the heatmap.
    """
    corr = df.corr(method=method)

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


# TODO: not tryed
def biplot(pca, dat):
    """
    Plot a data biplot on the screen
    :param dat: DataFrame data to plot
    :return:
    """
    print("computing biplot ...")
    # 0,1 denote PC1 and PC2; change values for other PCs
    xvector = pca.components_[0]  # see 'prcomp(my_data)$rotation' in R
    yvector = pca.components_[1]

    xs = pca.transform(dat)[:, 0]  # see 'prcomp(my_data)$x' in R
    ys = pca.transform(dat)[:, 1]

    # visualize projections

    # Note: scale values for arrows and text are a bit inelegant as of now,
    #       so feel free to play around with them

    # plt.figure(1)
    for i in range(len(xs)):
        # circles project documents (ie rows from csv) as points onto PC axes
        plt.plot(xs[i], ys[i], 'b,')
        # plt.text(xs[i] * 1.2, ys[i] * 1.2, list(dat.index)[i], color='b')

    for i in range(len(xvector)):
        # arrows project features (ie columns from csv) as vectors onto PC axes
        plt.arrow(0, 0, xvector[i] * max(xs), yvector[i] * max(ys),
                  color='r', width=0.0005, head_width=0.0025)
        plt.text(xvector[i] * max(xs) * 1.2, yvector[i] * max(ys) * 1.2,
                 list(dat.columns.values)[i], color='r')

    plt.show()
