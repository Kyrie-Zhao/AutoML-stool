import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
tf.reset_default_graph()
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import tensorflow.contrib.eager as tfe

import time
import argparse
import random
import string
import datetime
import cv2

from MODEL.VGG import B_VGGNet
from MODEL.loss import *
from MODEL.utils import read_all_batches, read_val_data
from MODEL.augementation import *
from MODEL.misc import progress_bar

class Solver_Train(object):

    def __init__(self, position=[16, 16, 16, 16]):
        self.position = position
        self.data_root = './cifar-10-batches-py'
        self.num_class = 10
        self.input_w = 32
        self.input_h = 32
        self.input_c = 3
        # Training parameter
        self.train_batch_size = 128
        self.test_batch_size = 1
        self.lr = 0.001
        self.momentum = 0.9
        self.epochs = 1
        self.train_step = None
        self.test_step = None
        self.checkpoint_path = "./model_checkpoints"

    def pretrain_load_data(self):
        train_data, train_label = read_all_batches(self.data_root, 5, [self.input_w, self.input_h, self.input_c])
        self.train_step = len(train_data) // self.train_batch_size
        val_data, val_label = read_val_data(self.data_root, [self.input_w, self.input_h, self.input_c], shuffle=False)
        self.test_step = len(val_data) // self.test_batch_size
        return (train_data, train_label), (val_data, val_label)

    def pretrain_get_augment_train_batch(self, train_data, train_labels, train_batch_size):
        '''
        This function helps generate a batch of train data, and random crop, horizontally flip
        and whiten them at the same time
        :param train_data: 4D numpy array
        :param train_labels: 1D numpy array
        :param train_batch_size: int
        :return: augmented train batch data and labels. 4D numpy array and 1D numpy array
        '''
        offset = np.random.choice(10000 - train_batch_size, 1)[0]
        batch_data = train_data[offset:offset+train_batch_size, ...]
        batch_data = random_crop_and_flip(batch_data, padding_size=2)
        batch_data = whitening_image(batch_data)
        batch_label = train_labels[offset:offset+self.train_batch_size]
        return batch_data, batch_label
    
    def pretrain_get_val_batch(self, vali_data, vali_label, vali_batch_size):
        '''
        If you want to use a random batch of validation data to validate instead of using the
        whole validation data, this function helps you generate that batch
        :param vali_data: 4D numpy array
        :param vali_label: 1D numpy array
        :param vali_batch_size: int
        :return: 4D numpy array and 1D numpy array
        '''
        offset = np.random.choice(10000 - vali_batch_size, 1)[0]
        vali_data_batch = vali_data[offset:offset+vali_batch_size, ...]
        vali_data_batch = whitening_image(vali_data_batch)
        vali_label_batch = vali_label[offset:offset+vali_batch_size]
        return vali_data_batch, vali_label_batch


    def pretrain(self,action):
        self.img_placeholder = tf.placeholder(dtype=tf.float32,
                                              shape=[self.train_batch_size, self.input_w, self.input_h, self.input_c],
                                              name='image_placeholder')
        self.label_placeholder = tf.placeholder(dtype=tf.int32,
                                                shape=[self.train_batch_size],
                                                name='label_placeholder')
        action_coarse = action[0]
        self.training_flag = tf.placeholder(dtype=tf.bool, shape=[], name='training_flag')

        self.B_VGG_instance = B_VGGNet(self.num_class, action)
        [logits_exit0, logits_exit1, logits_exit2, logits_exit3] = self.B_VGG_instance.model(self.img_placeholder,
                                                                                             is_train=self.training_flag)
        pred0 = tf.nn.softmax(logits_exit0, name='pred_exit0')
        loss_exit0 = cross_entropy(logits_exit0, self.label_placeholder)
        opt_exit2 = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
        #opt = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=self.momentum)
        train_op = opt_exit2.minimize(loss_exit0)
        # accuracy from brach
        train_acc0 = top_k_error(pred0, self.label_placeholder, 1)

        """for var in tf.trainable_variables():
            print(var.name, var.get_shape())"""

        # Initialize model and create session
        init = tf.initialize_all_variables()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(init)

        # Construct saver
        saver = tf.train.Saver()
        print(' Start Training '.center(50, '-'))
        (train_data, train_label), _ = self.pretrain_load_data()
        best_test_acc = 0
        for epoch in range(1, self.epochs + 1): 
            for train_step_num in range(1, self.train_step+1):
                step_list = []
                train_error_list = []
                total_loss_list = []
                branch_loss_list = []
                val_error_list = []
                train_data_batch, train_label_batch = self.pretrain_get_augment_train_batch(train_data, train_label, 
                                                                                            self.train_batch_size)
                _, train_loss, train_error0 = sess.run([train_op, loss_exit0, train_acc0],
                                                       feed_dict={self.img_placeholder: train_data_batch,
                                                                  self.label_placeholder: train_label_batch,
                                                                  self.training_flag: True})
                train_error_list.append(train_error0)
                total_loss_list.append(train_loss)
            print("[Epoch=%d]\tLoss=%.4f\tAccuracy=%.4f" % (epoch, np.mean(total_loss_list), np.mean(train_error_list)))
        save_path = saver.save(sess, os.path.join(self.checkpoint_path, 'pre_train.ckpt'))
        sess.close()
        print("Pre-Train End")
        
    def test(self, action):
        tf.reset_default_graph()
        coarse_num = 0
        coarse_crrect = 0
        sess = tf.Session()
        img_placeholder = tf.placeholder(dtype=tf.float32,
                                         shape=[self.test_batch_size, self.input_w, self.input_h, self.input_c],
                                         name='image_placeholder')
        label_placeholder = tf.placeholder(dtype=tf.int32,
                                           shape=[self.test_batch_size],
                                           name='label_placeholder')
        training_flag = tf.placeholder(dtype=tf.bool, shape=[], name='training_flag')

        self.B_VGG_instance = B_VGGNet(self.num_class,action)
        [logits_exit0, logits_exit1, logits_exit2, logits_exit3] = self.B_VGG_instance.model(img_placeholder,
                                                                                             is_train=training_flag)
        action_coarse = action[0]

        pred0 = tf.nn.softmax(logits_exit0, name='pred_exit0')

        train_acc0 = top_k_error(pred0, label_placeholder, 1)
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(self.checkpoint_path, 'pre_train.ckpt'))
        _, (val_data, val_label) = self.pretrain_load_data()
        label_list = []
        pred_list = []
        for i in range(self.test_step):
            if i%100==0:
                print(str(i)+" samples tested")
            test_data_batch, test_label_batch = self.pretrain_get_val_batch(val_data, val_label, self.test_batch_size)
            coarse_point, exit0_pred, test_acc0 = sess.run([self.B_VGG_instance.convertPosition[action_coarse],
                                                            pred0, train_acc0],
                                                           feed_dict={img_placeholder: test_data_batch,
                                                                      label_placeholder: test_label_batch,
                                                                      training_flag: False})

            coarse_num+=1
            if (test_acc0 == 1):
                coarse_crrect+=1

        print('Pretrain Accuracy: {}% '.format(coarse_crrect/coarse_num))
        sess.close()
        
    def train(self):
        self.position = [16,16,16,16]
        self.pretrain(self.position)
#         self.test(self.position)

def get_parser():
    """
    Creates an argument parser.
    """
    parser = argparse.ArgumentParser(description='Branchy_VGG with Stool Image Dataset')
    # Training parameters
    parser.add_argument('--phase', default='train', type=str, help='Train model or test')
    parser.add_argument('--cuda', default='1', type=str, help='CUDA visible devices',)
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    solver = Solver_Train()
    solver.train()

if __name__ == '__main__':
    main()