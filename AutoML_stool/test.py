import unittest
from MODEL.Client.model_util.solver_train import *

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

    def test_data_loader_coarse(self):
        ds_train = self.solverTrain.load_data_coarse(is_train=True)
        for images, labels_b, labels_c, labels_bc in ds_train.take(1):
            self.assertTrue(images.shape == (self.solverTrain.train_batch_size, 256, 256, 3))
            self.assertTrue(labels_b.shape == (self.solverTrain.train_batch_size, ))
            self.assertTrue(labels_b[0] <= 6 and labels_b[0] >= 0)
            self.assertTrue(labels_c[0] <= 3 and labels_b[0] >= 0)
            self.assertTrue(labels_bc[0] <= 3 and labels_b[0] >= 0)
            
        ds_test = self.solverTrain.load_data_coarse(is_train=False)
        for images, labels_b, labels_c, labels_bc in ds_test.take(1):
            self.assertTrue(images.shape == (self.solverTrain.test_batch_size, 256, 256, 3))
            self.assertTrue(labels_b.shape == (self.solverTrain.test_batch_size, ))
            self.assertTrue(labels_b[0] <= 6 and labels_b[0] >= 0)
            self.assertTrue(labels_c[0] <= 3 and labels_b[0] >= 0)
            self.assertTrue(labels_bc[0] <= 3 and labels_b[0] >= 0)
            
    def helper_test_data_loader_fine(self, ds, label_b_true, label_c_true, label_bc_true):
        for images, labels_b, labels_c, labels_bc in ds.take(1):
            self.assertTrue(images.shape == (self.solverTrain.train_batch_size//7, 256, 256, 3))
            self.assertTrue(labels_b.shape == (self.solverTrain.train_batch_size//7, ))
            self.assertTrue(np.all(labels_b.numpy() == label_b_true))
            self.assertTrue(np.all(labels_c.numpy() == label_c_true))
            self.assertTrue(np.all(labels_bc.numpy() == label_bc_true))
                
    def test_data_loader_fine(self):
        ds_0, ds_1, ds_2, ds_3, ds_4, ds_5, ds_6 = self.solverTrain.load_data_fine()
        for epoch in range(1, 10):
            # balacned sampling: smaple x from each class. (default, x=2, batch size = 2*7)
            X_train_0, y_b_train_0, y_c_train_0, y_bc_train_0 = next(iter(ds_0))
            X_train_1, y_b_train_1, y_c_train_1, y_bc_train_1 = next(iter(ds_1))
            X_train_2, y_b_train_2, y_c_train_2, y_bc_train_2 = next(iter(ds_2))
            X_train_3, y_b_train_3, y_c_train_3, y_bc_train_3 = next(iter(ds_3))
            X_train_4, y_b_train_4, y_c_train_4, y_bc_train_4 = next(iter(ds_4))
            X_train_5, y_b_train_5, y_c_train_5, y_bc_train_5 = next(iter(ds_5))
            X_train_6, y_b_train_6, y_c_train_6, y_bc_train_6 = next(iter(ds_6))
            
            X_train = tf.concat((X_train_0, X_train_1, X_train_2, X_train_3, 
                                 X_train_4, X_train_5, X_train_6), 0)
            y_c_train = tf.concat((y_c_train_0, y_c_train_1, y_c_train_2, y_c_train_3, 
                                   y_c_train_4, y_c_train_5, y_c_train_6), 0)
            
            self.assertTrue(X_train.shape == (self.solverTrain.train_batch_size, 256, 256, 3))
            self.assertTrue(y_c_train.shape == (self.solverTrain.train_batch_size, ))
            
        for epoch in range(1, 10):
            # balacned sampling: smaple x from each class. (default, x=2, batch size = 2*7)
            X_train_0, y_b_train_0, y_c_train_0, y_bc_train_0 = next(iter(ds_0))
            X_train_1, y_b_train_1, y_c_train_1, y_bc_train_1 = next(iter(ds_1))
            X_train_2, y_b_train_2, y_c_train_2, y_bc_train_2 = next(iter(ds_2))
            X_train_3, y_b_train_3, y_c_train_3, y_bc_train_3 = next(iter(ds_3))
            X_train_4, y_b_train_4, y_c_train_4, y_bc_train_4 = next(iter(ds_4))
            X_train_5, y_b_train_5, y_c_train_5, y_bc_train_5 = next(iter(ds_5))
            X_train_6, y_b_train_6, y_c_train_6, y_bc_train_6 = next(iter(ds_6))
            
            X_train_f0  = tf.concat((X_train_0, X_train_1), 0)
            y_bc_train_f0 = tf.concat((y_bc_train_0, y_bc_train_1), 0)
            X_train_f1  = tf.concat((X_train_2, X_train_3, X_train_4), 0)
            y_bc_train_f1 = tf.concat((y_bc_train_2, y_bc_train_3, y_bc_train_4), 0)
            X_train_f2  = tf.concat((X_train_5, X_train_6), 0)
            y_bc_train_f2 = tf.concat((y_bc_train_5, y_bc_train_6), 0)
             
            self.assertTrue(X_train_f0.shape == (self.solverTrain.train_batch_size//7 * 2, 256, 256, 3))
            self.assertTrue(y_bc_train_f0.shape == (self.solverTrain.train_batch_size //7 * 2, ))
            self.assertTrue(X_train_f1.shape == (self.solverTrain.train_batch_size//7 * 3, 256, 256, 3))
            self.assertTrue(y_bc_train_f1.shape == (self.solverTrain.train_batch_size //7 * 3, ))
            self.assertTrue(X_train_f2.shape == (self.solverTrain.train_batch_size//7 * 2, 256, 256, 3))
            self.assertTrue(y_bc_train_f2.shape == (self.solverTrain.train_batch_size //7 * 2, ))
       
                  
        ### This part came be resource-consuming
#         self.assertTrue(len(list(ds_0)) == 4)
#         self.assertTrue(len(list(ds_1)) == 22)
#         self.assertTrue(len(list(ds_2)) == 180)
#         self.assertTrue(len(list(ds_3)) == 650)
#         self.assertTrue(len(list(ds_4)) == 310)
#         self.assertTrue(len(list(ds_5)) == 177)
#         self.assertTrue(len(list(ds_6)) == 20)

        self.helper_test_data_loader_fine(ds_0, 0, 0, 0)
        self.helper_test_data_loader_fine(ds_1, 1, 0, 1)
        self.helper_test_data_loader_fine(ds_2, 2, 1, 0)
        self.helper_test_data_loader_fine(ds_3, 3, 1, 1)
        self.helper_test_data_loader_fine(ds_4, 4, 1, 2)
        self.helper_test_data_loader_fine(ds_5, 5, 2, 0)
        self.helper_test_data_loader_fine(ds_6, 6, 2, 1)  
        



            
if __name__ == '__main__':
    unittest.main()


