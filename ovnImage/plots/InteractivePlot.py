from .MultiPlot import *


class InteractivePlot(MultiPlot):

    FIGSIZE = (15, 15)

    def __init__(self, n_img=5):
        plt.ion()
        super().__init__(n_img)

    def set_authomatic_loop(self, value, time):
        self.authomatic = value
        self.interval = time

    def multi(self, images, cmap="Greys"):
        super().multi(images, cmap)
        if self.authomatic:
            self.fig.canvas.start_event_loop(self.interval)
        else:
            while not self.fig.waitforbuttonpress(0):
                pass
