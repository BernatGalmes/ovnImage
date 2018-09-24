import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class InteractivePlot:
    def __init__(self, n_img=5):
        plt.ion()

        self.authomatic = False
        self.interval = 1

        self.__create_figure(n_img)

    def __create_figure(self, n_img, keep_unused_axes=False):
        self.n_img = n_img
        nrows = 2
        if (self.n_img % 2) == 0:
            columns = int(self.n_img / nrows)
        else:
            columns = int((self.n_img / nrows) + 1)
        self.fig, self._axes = plt.subplots(nrows=nrows, ncols=columns, figsize=(15, 15))
        self.axes = self._axes.flatten()[:self.n_img]
        self.unused_axes = self._axes.flatten()[n_img:]
        if not keep_unused_axes:
            for uaxes in self.unused_axes:
                uaxes.remove()
            self.unused_axes = []

        self.fig.canvas.set_window_title('Interaction figure')
        self.colorbars = []

    def _multiplot(self, images, nrows=2, cmap='Greys'):
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

            # ax = plt.subplot(nrows, columns, i + 1)
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
                # cax.clear()
                cb = self.fig.colorbar(imshow, cax=cax, orientation='vertical')
                self.colorbars.append(cb)

    def set_authomatic_loop(self, value, time):
        self.authomatic = value
        self.interval = time

    def multi(self, images, nrows=2, cmap="Greys"):
        if len(images) > len(self.axes):
            self.__create_figure(self.n_img)

        self._multiplot(images, nrows, cmap)
        if self.authomatic:
            self.fig.canvas.start_event_loop(self.interval)
        else:
            while not self.fig.waitforbuttonpress(0):
                pass

    def save_multiplot(self, filename, images, nrows=2, cmap="Greys"):
        self._multiplot(images, nrows, cmap)
        plt.savefig(filename, bbox_inches='tight')