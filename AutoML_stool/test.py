import unittest
from MODEL.Client.model_util.solver_train import *
import tensorflow as tf
tf.enable_eager_execution()

class TestSolverTrain(unittest.TestCase):
    def setUp(self):
        self.data_root = './dataset_v4'
        self.a_bound = [16, 16, 16, 16]
        self.action = np.clip([np.random.rand() * self.a_bound[0],
                              np.random.rand() * self.a_bound[1],
                              np.random.rand() * self.a_bound[2],
                              np.random.rand() * self.a_bound[3]],np.zeros_like(self.a_bound), self.a_bound)
        self.solverTrain = Solver_Train(self.action)
        self.m_health = np.ones((20, 40))

    def test_data_loader(self):
        ds_train = self.solverTrain.load_data(is_train=True)
        for images, labels in ds_train.take(1):
            self.assertTrue(images.shape == (self.solverTrain.train_batch_size, 256, 256, 3))
            self.assertTrue(labels.shape == (self.solverTrain.train_batch_size, ))
            self.assertTrue(labels[0] <= 6 and labels[0] >= 0 )
            
        ds_test = self.solverTrain.load_data(is_train=False)
        for images, labels in ds_test.take(1):
            self.assertTrue(images.shape == (self.solverTrain.test_batch_size, 256, 256, 3))
            self.assertTrue(labels.shape == (self.solverTrain.test_batch_size, ))
            self.assertTrue(labels[0] <= 6 and labels[0] >= 0 )

if __name__ == '__main__':
    unittest.main()


