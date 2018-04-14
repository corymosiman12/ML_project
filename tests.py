import sys
import numpy as np
import utils
import create_features as cf
import data
import scipy.ndimage as ndimage

import unittest


class TestTransform(unittest.TestCase):
    def setUp(self):
        velocity = data.load_data()
        self.y_train = utils.sparse_dict(velocity, 256, 0)
        self.X_train = dict()
        for key, value in self.y_train.items():
            self.X_train[key] = ndimage.gaussian_filter(value, sigma=1, mode='wrap', truncate=500)

    def test_transform_simple(self):
        """

        """
        x1, y1 = utils.transform_dict_for_nn(self.X_train, self.y_train, 9)
        x1 = x1[:, : 256*256].T
        y1 = y1[: 256*256].reshape(256*256, 1)

        x2 = cf.form_features(self.X_train)['u']
        y2 = cf.my_reshaper(self.y_train)['u']
        self.assertTrue(np.allclose(y1, y2, rtol=1e-08, atol=1e-12))
        self.assertTrue(np.allclose(x1, x2, rtol=1e-08, atol=1e-12))


runner = unittest.TextTestRunner(verbosity=2).run(TestTransform("test_transform_simple"))


