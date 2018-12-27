from .MultiPlot import *


class InteractivePlot(MultiPlot):

    FIGSIZE = (15, 15)

    def __init__(self, n_img=5):
        """
        Init the object with the number of images/plots that must contain each Figure.

        :param n_img: Number of images/plots of the figures
        """
        plt.ion()
        super().__init__(n_img)

        self.authomatic = False
        self.interval = 1

    def set_authomatic_loop(self, value, time):
        """
        Specify if the plots has to be updated automatically or manually

        :param value: Activate or deactivate the automatic loop
        :type value: bool

        :param time: Number of seconds to wait bet
        :type time: float
        :return:
        """
        self.authomatic = value
        self.interval = time

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
        super().multi(images, cmap, title)
        if self.authomatic:
            self.fig.canvas.start_event_loop(self.interval)
        else:
            while not self.fig.waitforbuttonpress(0):
                pass
