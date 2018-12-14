import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class MultiPlot:

    FIGSIZE = (15, 15)

    def __init__(self, n_img=5):
        self.authomatic = False
        self.interval = 1

        self.colorbars = []
        self.n_img = n_img
        if n_img == 1:
            self.fig = plt.Figure(self.FIGSIZE)
            self.axes = [plt.axes()]
        elif n_img > 1:
            self.__create_figure()
        else:
            raise Exception("Bad number of images")

    def __create_figure(self, keep_unused_axes=False):
        nrows = 2
        if (self.n_img % 2) == 0:
            columns = int(self.n_img / nrows)
        else:
            columns = int((self.n_img / nrows) + 1)
        self.fig, self._axes = plt.subplots(nrows=nrows, ncols=columns, figsize=self.FIGSIZE)
        self.axes = self._axes.flatten()[:self.n_img]
        self.unused_axes = self._axes.flatten()[self.n_img:]
        if not keep_unused_axes:
            for uaxes in self.unused_axes:
                uaxes.remove()
            self.unused_axes = []

        self.fig.canvas.set_window_title('Interaction figure')

    def _multiplot(self, images, cmap='Greys'):
        """
        Help to plot a multiple image figure
        :param images: list of dictionaries with structure:
            {
            "img": image to plot,
            "title": name of the plot,
            "colorbar": show colorbar to the image
            "cmap": image matplotlib colormap --> https://matplotlib.org/examples/color/colormaps_reference.html
            }
        :param filename: String|None
                            Path file Where save the figure,
                            None to plot in a window
        :return:
        """
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

    def set_authomatic_loop(self, value, time):
        self.authomatic = value
        self.interval = time

    def multi(self, images, cmap="Greys"):
        if len(images) > len(self.axes):
            self.__create_figure()

        self._multiplot(images, cmap)

    def save_multiplot(self, filename, images, cmap="Greys"):
        self._multiplot(images, cmap)
        plt.savefig(filename, bbox_inches='tight')