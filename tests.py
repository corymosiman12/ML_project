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

        velocity = data.load_data(dimension=3)
        self.y_train_3d = utils.sparse_dict(velocity, 64, 0)
        self.X_train_3d = dict()
        for key, value in self.y_train_3d.items():
            self.X_train_3d[key] = ndimage.gaussian_filter(value, sigma=1, mode='wrap', truncate=500)

    def test_compare_transform(self):
        """

        """
        x1, y1 = utils.transform_dict_for_nn(self.X_train, self.y_train, 9)
        x1 = x1[:, : 256*256].T
        y1 = y1[: 256*256].reshape(256*256, 1)

        x2 = cf.form_features(self.X_train)['u']
        y2 = cf.my_reshaper(self.y_train)['u']
        self.assertTrue(np.allclose(y1, y2, rtol=1e-08, atol=1e-12))
        self.assertTrue(np.allclose(x1, x2, rtol=1e-08, atol=1e-12))

    def test_transform_y(self):
        for n in [9, 27, 25]:
            x, y = utils.transform_dict_for_nn(self.X_train, self.y_train, n)
            y_train_new = utils.untransform_y(y, self.y_train['u'].shape)
            for key, value in self.y_train.items():
                self.assertTrue(np.allclose(self.y_train[key], y_train_new[key], rtol=1e-08, atol=1e-12))

    def test_transform_y_3D(self):
        x, y = utils.transform_dict_for_nn_3D(self.X_train_3d, self.y_train_3d, 27)
        y_train_new = utils.untransform_y_3D(y, self.y_train_3d['u'].shape)
        for key, value in self.y_train_3d.items():
            self.assertTrue(np.allclose(self.y_train_3d[key], y_train_new[key], rtol=1e-08, atol=1e-12))


# runner = unittest.TextTestRunner(verbosity=2).run(TestTransform("test_compare_transform"))
runner = unittest.TextTestRunner(verbosity=2).run(TestTransform("test_transform_y"))
runner = unittest.TextTestRunner(verbosity=2).run(TestTransform("test_transform_y_3D"))
