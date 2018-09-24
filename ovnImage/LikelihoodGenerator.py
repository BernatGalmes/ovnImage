import cv2
from .images import *
from .masks import mask_fill_holes


class LikelihoodGenerator:
    """
    Class to help to generate likelihood masks of regions of
    interest in images.

    Build your likelihood by marking with the mouse the contours of
    the roi region.

    Keyboard commands:
    key a: Submit region selected.
    keb b: discard last mouse click

    Example of use:
    >>> image = "RGB image matrix"
    >>> path_output_file = "Path to save the likelihood"
    >>> LikelihoodGenerator(image).build_your_mask(path_output_file)
    """

    def __init__(self, image):
        """
        Init an object to build the interest region of an image
        :param image: RGB image
        """
        self.image = image.copy()

        self.in_points = []  # etiquetes de les finger_regions seleccionades

    def __fill_event(self, event, x, y, flags, param):
        """
        Event to execute when click onto a region
        :param event:
        :param x:
        :param y:
        :param flags:
        :param param:
        :return:
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            point = (x, y)
            print("event at position: " + str(point))

            cv2.circle(self.image, point, 5, (0, 0, 0), -1)

            if len(self.in_points) != 0:
                cv2.line(self.image, self.in_points[-1], point, (0, 0, 0))

            self.in_points.append(point)

            cv2.destroyWindow("image")

    def build_your_mask(self, file=None):
        """
        Call this function to init the process of finger_regions selection.
        In the process:
        Press A (shift + a) to end selection
        Press b to cancel the last selection
        :return:
        """
        key = 0
        while "a" != chr(key & 255):
            cv2.namedWindow("image", flags=cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("image", self.__fill_event)
            cv2.resizeWindow('image', 600, 600)

            # display the image and wait for a keypress
            cv2.imshow("image", self.image)
            key = cv2.waitKey(-1)
            print("pressed: " + str(chr(key & 255)))

            if "b" == chr(key & 255):
                value = self.in_points.pop()
                cv2.line(self.image, self.in_points[-1], value, (255, 255, 255))

        # close all open windows
        cv2.destroyAllWindows()

        mask = np.zeros(self.image.shape[0:2], dtype=np.uint8)
        p1 = self.in_points[0]
        for idx in range(1, len(self.in_points)):
            p2 = self.in_points[idx]
            cv2.line(mask, p1, p2, 255, 1)
            p1 = p2

        mask *= 255
        mask = mask_fill_holes(mask)
        if file is not None:
            print("Writting file to: " + str(file))
            # TODO: check if folder exists, if not create dir
            cv2.imwrite(file, binary2RGB(mask))

        return mask
