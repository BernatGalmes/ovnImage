import unittest
import cv2
from bern_img_utils.masks import *


class MasksTestCase(unittest.TestCase):

    def setUp(self):
        self.mask = mask_from_RGB_file("../data/mask.jpg")
        self.likelihood = mask_from_RGB_file("../data/likelihood.jpg")
        self.teoric_evaluation = {'cohen_kappa': 0.54313501884871074,
                                  'Recall': 0.5912653532924077,
                                  'FN': 0.4087346467075923,
                                  'Precision': 0.9986451676658918,
                                  'accuracy': 0.8484160335303632,
                                  'TP': 0.5912653532924077,
                                  'FP': 0.0008021522004164241,
                                  'F1': 0.7427641745748484}

        self.teoric_sklearn_evaluation = {'cohen_kappa': 0.54313501884871074,
                                          'F1': 0.75953388296686275,
                                          'Recall': 0.75953388296686275,
                                          'Fbeta': 0.75953388296686275,
                                          'accuracy': 0.75953388296686275,
                                          'Precision': 0.75953388296686275}


    def test_evaluation(self):
        result = mask_evaluation(self.mask, self.likelihood)
        self.assertEqual(result, self.teoric_evaluation)

    def test_sklearn_evaluation(self):
        result = mask_sklearn_evaluation(self.mask, self.likelihood)

        self.assertEqual(result, self.teoric_sklearn_evaluation)

    def test_coincidence(self):
        bool_mask = self.mask.copy()
        true_indexs = np.nonzero(bool_mask)
        true_indexs = np.array(true_indexs)
        remove_indexs = (true_indexs[0][0:int(len(true_indexs[0]) * 0.1)], true_indexs[1][0:int(len(true_indexs[0]) * 0.1)])
        mask = bool_mask.copy()
        mask[remove_indexs] = 0

        self.assertEqual(round(masks_coincidence(bool_mask, mask), 2), 0.9)

    def test_onto_mask(self):

        self.assertEqual(mask_onto_mask(self.mask, self.mask), True)

    # def test_2RGB(self):
    #     self.assertEqual(True, False)

    def test_fill_holes(self):
        mask_ones = self.likelihood.copy()
        _, contours, _ = cv2.findContours(mask_ones.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                M = cv2.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(mask_ones, center=(cx, cy), radius=2, color=0)

        mask_ones_save = mask_ones.copy()
        mask = mask_fill_holes(mask_ones)
        self.assertEqual(masks_coincidence(self.likelihood, mask) > 0.99, True)

        # mask ones must not be modified
        self.assertEqual(masks_coincidence(mask_ones, mask_ones_save) > 0.99, True)

        mask_255 = mask_ones * 255
        mask = mask_fill_holes(mask_255)
        self.assertEqual(masks_coincidence(self.likelihood, mask) > 0.99, True)

    def test_from_RGB_file(self):
        mask = mask_from_RGB_file("../data/likelihood.jpg")
        self.assertEqual(np.max(mask), 1)
        self.assertEqual(np.min(mask), 0)

    def test_bounding_circle(self):
        center, r = mask_bounding_circle(self.mask)
        self.assertEqual(center, (314, 274))
        self.assertEqual(r, 415)

    # def test_delete_contour_in(self):
    #     self.assertEqual(True, False)

    # def test_build_circular(self):
    #     self.assertEqual(True, False)
    #
    # def test_build_circular_boolean(self):
    #     self.assertEqual(True, False)

    def test_biggest_connected_component(self):
        mask_drawed = self.likelihood.copy()
        h, w = mask_drawed.shape[:2]
        points = np.array([
            (0, 0),
            (0, 100),
            (100, 0)
        ])
        cv2.fillConvexPoly(mask_drawed, points, 1)
        points = np.array([
            (0, w),
            (0, w - 100),
            (100, w)

        ])
        cv2.fillConvexPoly(mask_drawed, points, 1)

        mask_big_comp = mask_biggest_connected_component(mask_drawed)

        # Check if drawed mask hasn't been modified with function
        self.assertEqual(masks_coincidence(self.likelihood, mask_drawed) < 0.98, True)

        # Check if biggest connected component is equal than the likelihood
        self.assertEqual(masks_coincidence(self.likelihood, mask_big_comp) > 0.99, True)


if __name__ == '__main__':
    unittest.main()
