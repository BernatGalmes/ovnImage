import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class MultiPlot:
    """
    Class to help to plot a set of images or plots in a unique figure.

    """

    FIGSIZE = (15, 15)

    def __init__(self, n_img=5, n_rows=2):
        """
        Init the object with the number of images/plots that must contain each Figure.

        :param n_img: Number of images/plots of the figures
        """
        self.colorbars = []
        self.n_img = n_img
        self.n_rows = n_rows
        if n_img == 1:
            self.fig = plt.Figure(self.FIGSIZE)
            self.axes = [plt.axes()]
        elif n_img > 1:
            self.__create_figure()
        else:
            raise Exception("Bad number of images")

    def __create_figure(self, keep_unused_axes=False):
        """
        Internal function that build the Figure that are going to contain
        all the plots

        :param keep_unused_axes: Specify if the axes of the subplots must be keep
        :type keep_unused_axes: bool

        :return:
        """
        if (self.n_img % 2) == 0:
            columns = int(self.n_img / self.n_rows)
        else:
            columns = int((self.n_img / self.n_rows) + 1)
        self.fig, self._axes = plt.subplots(nrows=self.n_rows, ncols=columns, figsize=self.FIGSIZE)
        self.axes = self._axes.flatten()[:self.n_img]
        self.unused_axes = self._axes.flatten()[self.n_img:]
        if not keep_unused_axes:
            for uaxes in self.unused_axes:
                uaxes.remove()
            self.unused_axes = []

        self.fig.canvas.set_window_title('Interaction figure')

    def _multiplot(self, images, cmap='Greys', title=None):
        """
        Help to plot a multiple image figure
        :param images: list of dictionaries with structure:
            {
            "img": image to plot,
            "title": name of the plot,
            "colorbar": show colorbar to the image
            "cmap": image matplotlib colormap --> https://matplotlib.org/examples/color/colormaps_reference.html
            }
        :param cmap: Default color map to use in all subplots
        :param title: Main title to show in the figure.

        :return:
        """
        if title is not None:
            self.fig.suptitle(title, fontsize=16)

        for cb in self.colorbars:
            cb.remove()
        self.colorbars = []
        for i, img in enumerate(images):
            ax = self.axes[i]

            image = img['img']
            title = img['title']

            ax.clear()
            ax.set_title(title)
            ax.axis('off')
            if 'cmap' in img:
                imshow = ax.imshow(image, cmap=img['cmap'])

            else:
                imshow = ax.imshow(image, cmap=cmap)

            if 'colorbar' in img:
                im = ax.images  # this is a list of all images that have been plotted
                cb = im[-1].colorbar
                if cb is not None:
                    cb.remove()
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cb = self.fig.colorbar(imshow, cax=cax, orientation='vertical')
                self.colorbars.append(cb)

    def update(self, images, cmap="Greys"):
        return self._multiplot(images, cmap)

    def multi(self, images, cmap="Greys", title=None):
        """
        Plot a multiple image figure in a window.

        :param images: list of dictionaries with structure:
            {
            "img": image to plot,
            "title": name of the plot,
            "colorbar": show colorbar to the image
            "cmap": image matplotlib colormap --> https://matplotlib.org/examples/color/colormaps_reference.html
            }
        :param cmap: Default color map to use in all subplots
        :param title: Main title to show in the figure.

        :return:
        """
        if len(images) > len(self.axes):
            self.__create_figure()

        self._multiplot(images, cmap, title)

    def save_multiplot(self, filename, images, cmap="Greys"):
        """
        Plot a multiple image figure in a file.

        :param filename: String|None
                            Path file Where save the figure,
                            None to plot in a window

        :param images: list of dictionaries with structure:
            {
            "img": image to plot,
            "title": name of the plot,
            "colorbar": show colorbar to the image
            "cmap": image matplotlib colormap --> https://matplotlib.org/examples/color/colormaps_reference.html
            }

        :param cmap: Default color map to use in all subplots

        :return:
        """
        self._multiplot(images, cmap)
        plt.savefig(filename, bbox_inches='tight')

    def attach_key_press_event(self, function):
        self.fig.canvas.mpl_connect("key_press_event", function)

