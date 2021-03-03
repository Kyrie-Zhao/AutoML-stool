import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import tensorflow as tf
# tf.enable_eager_execution()
# tf.compat.v1.disable_eager_execution() # what for?
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import tensorflow.contrib.eager as tfe

import time

import argparse
import random
import string
import datetime
import cv2

# from VGG import B_VGGNet
# from loss import *
# from utils import read_all_batches, read_val_data, write_pickle
# from misc import progress_bar
# from augementation import *

from MODEL.Client.model_util.VGG import B_VGGNet
from MODEL.Client.model_util.loss import *
from MODEL.Client.model_util.utils import read_all_batches, read_val_data, write_pickle
from MODEL.Client.model_util.misc import progress_bar
from MODEL.Client.model_util.augementation import *


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#from AlexNet import B_AlexNet
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

MAX_SIZE = 1299

class Solver_Train(object):

    def __init__(self, position=[0, 0, 0, 0]):
        #self.data_root = './data/cifar-10-python/cifar-10-batches-py'
        self.position = position
        self.data_root = './dataset_v4'
        self.num_class = 3
        self.input_w = 28
        self.input_h = 28
        self.input_c = 3
        # Training parameter
        self.position = position
        self.train_batch_size = 7
        self.test_batch_size = 1
        self.lr = 0.001
        self.momentum = 0.9
        self.epochs = int(10 * (MAX_SIZE/self.train_batch_size*7))
        self.baseline_epoch = 30
        self.train_step = None
        self.test_step = None
        self.checkpoint_path = "./model_checkpoints"
        self.auto_earlyexit_lossweights = [0.03810899993000662, 0.017216535550954308, 0.05113813155702535, 0.022274152735761877]
        print(self.auto_earlyexit_lossweights)

        self.train_branch_acc = {}
        self.test_branch_acc = {}
        self.inference_time = {}

        self.earlyexit_lossweights = [1.0, 0.0, 0.0, 0.0]
        self.earlyexit_thresholds = [0,0,10]
        for i in range(len(self.earlyexit_lossweights)):
            self.inference_time['exit%d'%(i)] = []
            self.train_branch_acc['exit%d'%(i)] = []
            self.test_branch_acc['exit%d'%(i)] = []

    def augment(self, image, labels_b, labels_c, labels_bc):
        image = tf.image.resize_images(image, size=[self.input_w, self.input_h])
        return image, labels_b, labels_c, labels_bc

    def read_image(self, image_file, labels_b, labels_c, labels_bc):
        directory = self.data_root
        image = tf.io.read_file(directory +'/' +image_file)
        image = tf.image.decode_png(image, channels=3)
        return image, labels_b, labels_c, labels_bc

    def load_data_train_helper(self, df, bristol):
        index_bristol = df.loc[:, 'bristol_type']==bristol
        paths_file = df.loc[index_bristol, 'image_id'].values
        labels_b = df.loc[index_bristol, 'bristol_type'].values
        labels_c = df.loc[index_bristol, 'condition'].values
        labels_bc = df.loc[index_bristol, 'brsitol_on_condition'].values
        ds = tf.data.Dataset.from_tensor_slices((paths_file, labels_b, labels_c, labels_bc)).repeat(-1)
        ds = ds.map(self.read_image).map(self.augment).batch(self.train_batch_size // 7)
        return ds

    def load_data_train(self):
        directory = self.data_root
        df = pd.read_csv(os.path.join(directory, 'train_a_b_annotation.csv'))
        max_size = max(df.loc[:, 'bristol_type'].value_counts())
        ds_0 = self.load_data_train_helper(df, 0)
        ds_1 = self.load_data_train_helper(df, 1)
        ds_2 = self.load_data_train_helper(df, 2)
        ds_3 = self.load_data_train_helper(df, 3)
        ds_4 = self.load_data_train_helper(df, 4)
        ds_5 = self.load_data_train_helper(df, 5)
        ds_6 = self.load_data_train_helper(df, 6)
        return ds_0, ds_1, ds_2, ds_3, ds_4, ds_5, ds_6

    def load_data_test(self, test_baseline = 'a'):
        directory = self.data_root
        if test_baseline == 'a':
            df = pd.read_csv(os.path.join(directory, 'test_a_annotation.csv'))
        else:
            df = pd.read_csv(os.path.join(directory, 'test_b_annotation.csv'))
        paths_file = df.loc[:, 'image_id'].values
        labels_b = df.loc[:, 'bristol_type'].values
        labels_c = df.loc[:, 'condition'].values
        labels_bc = df.loc[:, 'brsitol_on_condition'].values
        ds = tf.data.Dataset.from_tensor_slices((paths_file, labels_b, labels_c, labels_bc))
        ds = ds.map(self.read_image).map(self.augment).batch(self.test_batch_size)
        return ds
    

    def train_coarse(self,position):
        # create placeholder
        self.img_placeholder = tf.placeholder(dtype=tf.float32,
                                              shape=[self.train_batch_size, self.input_w, self.input_h, self.input_c],
                                              name='image_placeholder')
        self.label_placeholder = tf.placeholder(dtype=tf.int32,
                                                shape=[self.train_batch_size],
                                                name='label_placeholder')
        self.training_flag = tf.placeholder(dtype=tf.bool, shape=[], name='training_flag')

        self.B_VGG_instance = B_VGGNet(num_class=self.num_class)
        [logits_exit0, logits_exit1, logits_exit2, logits_exit3] = self.B_VGG_instance.model(self.img_placeholder, # 2 F1
                                                                               is_train=self.training_flag)
        print('VGG DONE')
        # prediction from branches
        pred0 = tf.nn.softmax(logits_exit0, name='pred_exit0')

        # logits of branches
        loss_exit0 = cross_entropy(logits_exit0, self.label_placeholder)
        opt_exit2 = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
        #opt = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=self.momentum)
        train_op = opt_exit2.minimize(loss_exit0)
        # accuracy from brach
        train_acc0 = top_k_error(pred0, self.label_placeholder, 1)

        for var in tf.trainable_variables():
            print(var.name, var.get_shape())

        # Initialize model and create session
        init = tf.initialize_all_variables()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        #log_device_placement=True
        sess.run(init)

        # Construct saver
        saver = tf.train.Saver()
        print(' Start Training '.center(50, '-'))

        ds_train_0, ds_train_1, ds_train_2, ds_train_3, ds_train_4, ds_train_5, ds_train_6 = self.load_data_train()
        batch_0 = ds_train_0.make_one_shot_iterator().get_next()
        batch_1 = ds_train_1.make_one_shot_iterator().get_next()
        batch_2 = ds_train_2.make_one_shot_iterator().get_next()
        batch_3 = ds_train_3.make_one_shot_iterator().get_next()
        batch_4 = ds_train_4.make_one_shot_iterator().get_next()
        batch_5 = ds_train_5.make_one_shot_iterator().get_next()
        batch_6 = ds_train_6.make_one_shot_iterator().get_next()
        
        for epoch in range(1, self.epochs + 1): # self.epochs = 10 * (MAX_SIZE/2)
            step_list = []
            train_error_list = []
            total_loss_list = []
            branch_loss_list = []
            val_error_list = []
            # balacned sampling: smaple x from each class. (default, x=2, batch size = 2*7)
            X_train_0, y_b_train_0, y_c_train_0, y_bc_train_0 = sess.run(batch_0)
            X_train_1, y_b_train_1, y_c_train_1, y_bc_train_1 = sess.run(batch_1)
            X_train_2, y_b_train_2, y_c_train_2, y_bc_train_2 = sess.run(batch_2)
            X_train_3, y_b_train_3, y_c_train_3, y_bc_train_3 = sess.run(batch_3)
            X_train_4, y_b_train_4, y_c_train_4, y_bc_train_4 = sess.run(batch_4)
            X_train_5, y_b_train_5, y_c_train_5, y_bc_train_5 = sess.run(batch_5)
            X_train_6, y_b_train_6, y_c_train_6, y_bc_train_6 = sess.run(batch_6)

            X_train = tf.concat((X_train_0, X_train_1, X_train_2, X_train_3,
                                 X_train_4, X_train_5, X_train_6), 0)
            y_c_train = tf.concat((y_c_train_0, y_c_train_1, y_c_train_2, y_c_train_3,
                                   y_c_train_4, y_c_train_5, y_c_train_6), 0)
            #(14, 256, 256, 3)

            _, train_loss, train_error0 = sess.run([train_op, total_loss, train_acc0], 
                                                   feed_dict={self.img_placeholder: X_train, 
                                                              self.label_placeholder: y_c_train,
                                                              self.training_flag: True})
            train_error_list.append([train_error0])
            total_loss_list.append(train_loss)
            print('Loss: {:.4f}'.format(train_loss))
            #print('Acc: {:.4f} | {:.4f} | {:.4f}| {:.4f}'.format(train_error0, train_error1, train_error2, train_error3))
            progress_bar(epoch, self.epochs + 1)

        save_path = saver.save(sess, os.path.join(self.checkpoint_path, 'coarse.ckpt'))
        sess.close()

    def train_fine(self,position):
        #Action
        convertPosition=[self.B_VGG_instance.conv1, self.B_VGG_instance.conv2, self.B_VGG_instance.max_pool1,
                         self.B_VGG_instance.conv3, self.B_VGG_instance.conv4, self.B_VGG_instance.max_pool2,
                         self.B_VGG_instance.conv5, self.B_VGG_instance.conv6, self.B_VGG_instance.conv7,
                         self.B_VGG_instance.max_pool3, self.B_VGG_instance.conv8, self.B_VGG_instance.conv9,
                         self.B_VGG_instance.conv10, self.B_VGG_instance.max_pool4, self.B_VGG_instance.conv11,
                         self.B_VGG_instance.conv12, self.B_VGG_instance.conv13]
        action_coarse = position[0]
        action_fine_1 = position[1]
        action_fine_2 = position[2]
        action_fine_3 = position[3]
        # create placeholder
        self.img_placeholder = tf.placeholder(dtype=tf.float32,
                                              shape=[self.test_batch_size, self.input_w, self.input_h, self.input_c],
                                              name='image_placeholder')
        self.label_placeholder_1 = tf.placeholder(dtype=tf.int32,
                                                shape=[self.test_batch_size],
                                                name='label_placeholder')
        self.label_placeholder_2 = tf.placeholder(dtype=tf.int32,
                                                shape=[self.test_batch_size],
                                                name='label_placeholder')
        self.label_placeholder_3 = tf.placeholder(dtype=tf.int32,
                                                shape=[self.test_batch_size],
                                                name='label_placeholder')
        self.training_flag = tf.placeholder(dtype=tf.bool, shape=[], name='training_flag')
        self.earlyexit_lossweights_placeholder = tf.placeholder(dtype=tf.float32,
                                                                shape=[3],
                                                                name='earlyexit_lossweights_placeholder')
        # create model and build graph
        self.B_VGG_instance = B_VGGNet(num_class=self.num_class)
        [logits_exit0, logits_exit1, logits_exit2, logits_exit3] = self.B_VGG_instance.model(self.img_placeholder,
                                                                                             is_train=self.training_flag)

        # prediction from branches
        pred1 = tf.nn.softmax(logits_exit1, name='pred_exit1')
        pred2 = tf.nn.softmax(logits_exit2, name='pred_exit2')
        pred3 = tf.nn.softmax(logits_exit3, name='pred_exit3')

        loss_exit1 = cross_entropy(logits_exit1, self.label_placeholder_1)
        loss_exit2 = cross_entropy(logits_exit2, self.label_placeholder_2)
        loss_exit3 = cross_entropy(logits_exit3, self.label_placeholder_3)
        total_loss = tf.reduce_sum(tf.multiply(self.earlyexit_lossweights_placeholder,
                                               [loss_exit1, loss_exit2, loss_exit3]))
        # logits of branches
        # accuracy from brach
        opt_exit2 = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
        train_op = opt_exit2.minimize(total_loss)

        train_acc1 = top_k_error(pred1, self.label_placeholder, 1)
        train_acc2 = top_k_error(pred2, self.label_placeholder, 1)
        train_acc3 = top_k_error(pred3, self.label_placeholder, 1)

        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(self.checkpoint_path, 'coarse.ckpt'))
        ds_train_0, ds_train_1, ds_train_2, ds_train_3, ds_train_4, ds_train_5, ds_train_6 = self.load_data_train()
        batch_0 = ds_train_0.make_one_shot_iterator().get_next()
        batch_1 = ds_train_1.make_one_shot_iterator().get_next()
        batch_2 = ds_train_2.make_one_shot_iterator().get_next()
        batch_3 = ds_train_3.make_one_shot_iterator().get_next()
        batch_4 = ds_train_4.make_one_shot_iterator().get_next()
        batch_5 = ds_train_5.make_one_shot_iterator().get_next()
        batch_6 = ds_train_6.make_one_shot_iterator().get_next()
        
        for epoch in range(1, self.epochs + 1):
            # balacned sampling: smaple x from each class. (default, x=2, batch size = 2*7)
            X_train_0, y_b_train_0, y_c_train_0, y_bc_train_0 = sess.run(batch_0)
            X_train_1, y_b_train_1, y_c_train_1, y_bc_train_1 = sess.run(batch_1)
            X_train_2, y_b_train_2, y_c_train_2, y_bc_train_2 = sess.run(batch_2)
            X_train_3, y_b_train_3, y_c_train_3, y_bc_train_3 = sess.run(batch_3)
            X_train_4, y_b_train_4, y_c_train_4, y_bc_train_4 = sess.run(batch_4)
            X_train_5, y_b_train_5, y_c_train_5, y_bc_train_5 = sess.run(batch_5)
            X_train_6, y_b_train_6, y_c_train_6, y_bc_train_6 = sess.run(batch_6)

            X_train_f0  = tf.concat((X_train_0, X_train_1), 0)
            y_bc_train_f0 = tf.concat((y_bc_train_0, y_bc_train_1), 0) # 0, 1
            X_train_f1  = tf.concat((X_train_2, X_train_3, X_train_4), 0)
            y_bc_train_f1 = tf.concat((y_bc_train_2, y_bc_train_3, y_bc_train_4), 0) # 0, 1, 2
            X_train_f2  = tf.concat((X_train_5, X_train_6), 0)
            y_bc_train_f2 = tf.concat((y_bc_train_5, y_bc_train_6), 0) # 0, 1

            #FINE1
            if(action_fine_1<=action_coarse):
                fine1_out = sess.run([convertPosition[action_fine_1]], 
                                     feed_dict={self.img_placeholder: X_train_f0,
                                                self.label_placeholder: y_bc_train_f0,
                                                self.training_flag: False})
                _, train_loss, train_error1 = sess.run([train_op, total_loss, train_acc1],
                                                       feed_dict={self.img_placeholder: fine1_out,
                                                                  self.label_placeholder: y_bc_train_f0, 
                                                                  self.earlyexit_lossweights_placeholder: [0.3,0.4,0.3], 
                                                                  self.training_flag: True})
            else:
                coarse_out = sess.run([convertPosition[action_coarse]], 
                                      feed_dict={self.img_placeholder: X_train_f0,
                                                 self.label_placeholder: y_bc_train_f0,
                                                 self.training_flag: False})
                _, train_loss, train_error1 = sess.run([train_op, total_loss, train_acc1],
                                                       feed_dict={self.img_placeholder: coarse_out,
                                                                  self.label_placeholder: y_bc_train_f0,
                                                                  self.earlyexit_lossweights_placeholder: [0.3,0.4,0.3],
                                                                  self.training_flag: True})
            if(action_fine_2<=action_coarse):
                fine2_out = sess.run([convertPosition[action_fine_2]], feed_dict={self.img_placeholder: X_train_f1,
                                                                                  self.label_placeholder: y_bc_train_f1,
                                                                                  self.training_flag: False})
                _, train_loss, train_error2 = sess.run([train_op, total_loss, train_acc2],
                                                       feed_dict={self.img_placeholder: fine2_out,
                                                                  self.label_placeholder: y_bc_train_f1,
                                                                  self.earlyexit_lossweights_placeholder: [0.3,0.4,0.3],
                                                                  self.training_flag: True})
            else:
                coarse_out = sess.run([convertPosition[action_coarse]], feed_dict={self.img_placeholder: X_train_f1,
                                                                                   self.label_placeholder: y_bc_train_f1,
                                                                                   self.training_flag: False})
                _, train_loss, train_error2 = sess.run([train_op, total_loss, train_acc2],
                                                       feed_dict={self.img_placeholder: coarse_out,
                                                                  self.label_placeholder: y_bc_train_f1,
                                                                  self.earlyexit_lossweights_placeholder: [0.3,0.4,0.3],
                                                                  self.training_flag: True})
            if(action_fine_3<=action_coarse):
                fine3_out = sess.run([convertPosition[action_fine_3]], feed_dict={self.img_placeholder: X_train_f2,
                                                                                  self.label_placeholder: y_bc_train_f2,
                                                                                  self.training_flag: False})
                _, train_loss, train_error3 = sess.run([train_op, total_loss, train_acc3],
                                                       feed_dict={self.img_placeholder: fine3_out,
                                                                  self.label_placeholder: y_bc_train_f2,
                                                                  self.earlyexit_lossweights_placeholder: [0.3,0.4,0.3],
                                                                  self.training_flag: True})
            else:
                coarse_out = sess.run([convertPosition[action_coarse]], 
                                      feed_dict={self.img_placeholder: X_train_f2,
                                                 self.label_placeholder: y_bc_train_f2,
                                                 self.training_flag: False})
                _, train_loss, train_error3 = sess.run([train_op, total_loss, train_acc3],
                                                       feed_dict={self.img_placeholder: coarse_out,
                                                                  self.label_placeholder: y_bc_train_f2,
                                                                  self.earlyexit_lossweights_placeholder: [0.3,0.4,0.3],
                                                                  self.training_flag: True})

            #train_error_list.append([train_error0])
            total_loss_list.append(train_loss)
            print('Loss: {:.4f}'.format(train_loss))
            progress_bar(epoch, self.epochs + 1)

        save_path = saver.save(sess, os.path.join(self.checkpoint_path, 'AutoML_stool.ckpt'))
        sess.close()

            #FINE2
            #FINE3

    def test(self):
        
        ds_test = self.load_data_test()
        for X_test, y_b_test, y_c_test, y_bc_test in tfe.Iterator(ds_test):

            pass


    def train(self):
        self.train_coarse(self.position)
        self.train_fine(self.position)

    def test(self,action):

        # create placeholder
        self.img_placeholder = tf.placeholder(dtype=tf.float32,
                                              shape=[self.test_batch_size, self.input_w, self.input_h, self.input_c],
                                              name='image_placeholder')
        self.label_placeholder = tf.placeholder(dtype=tf.int32,
                                                shape=[self.test_batch_size],
                                                name='label_placeholder')
        self.training_flag = tf.placeholder(dtype=tf.bool, shape=[], name='training_flag')
        self.earlyexit_lossweights_placeholder = tf.placeholder(dtype=tf.float32,
                                                                shape=[len(self.earlyexit_lossweights)],
                                                                name='earlyexit_lossweights_placeholder')
        # create model and build graph
        self.B_VGG_instance = B_VGGNet(num_class=self.num_class)
        [logits_exit0, logits_exit1, logits_exit2, logits_exit3] = self.B_VGG_instance.model(self.img_placeholder,
                                                                                             is_train=self.training_flag)

        # prediction from branches
        pred0 = tf.nn.softmax(logits_exit0, name='pred_exit0')
        pred1 = tf.nn.softmax(logits_exit1, name='pred_exit1')
        pred2 = tf.nn.softmax(logits_exit2, name='pred_exit2')
        pred3 = tf.nn.softmax(logits_exit3, name='pred_exit3')

        # logits of branches
        # accuracy from brach
        train_acc0 = top_k_error(pred0, self.label_placeholder, 1)
        train_acc1 = top_k_error(pred1, self.label_placeholder, 1)
        train_acc2 = top_k_error(pred2, self.label_placeholder, 1)
        train_acc3 = top_k_error(pred3, self.label_placeholder, 1)

        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(self.checkpoint_path, 'B_VGG.ckpt'))
        # load all data in memory
#         _, (val_data, val_label) = self.load_data()
        coarse_crrect = 0
        fine1_crrect = 0
        fine2_crrect = 0
        fine3_crrect = 0

        coarse_num = 0
        fine1_num = 0
        fine2_num = 0
        fine3_num = 0

        strWriter = ""
        # inference / test
        print(self.test_step)
        time_average_arrary = []
        acc_arrary = []

        #Action
        convertPosition=[self.B_VGG_instance.conv1, self.B_VGG_instance.conv2, self.B_VGG_instance.max_pool1,
                         self.B_VGG_instance.conv3, self.B_VGG_instance.conv4, self.B_VGG_instance.max_pool2,
                         self.B_VGG_instance.conv5, self.B_VGG_instance.conv6, self.B_VGG_instance.conv7,
                         self.B_VGG_instance.max_pool3, self.B_VGG_instance.conv8, self.B_VGG_instance.conv9,
                         self.B_VGG_instance.conv10, self.B_VGG_instance.max_pool4, self.B_VGG_instance.conv11,
                         self.B_VGG_instance.conv12, self.B_VGG_instance.conv13]
        action_coarse = action[0]
        action_fine_1 = action[1]
        action_fine_2 = action[2]
        action_fine_3 = action[3]

        ds_test = self.load_data(is_train=False)
        for test_data_batch, test_label_b_batch, test_label_c_batch, test_label_bc_batch in tfe.Iterator(ds_test):
            coarse_point, fine_1_point, fine_2_point, fine_3_point, \
            exit0_pred, test_acc0 = sess.run([convertPosition[action[0]], convertPosition[action[1]],
                                              convertPosition[action[2]], convertPosition[action[3]],
                                              pred0, train_acc0],
                                             feed_dict={self.img_placeholder: test_data_batch,
                                                        self.label_placeholder: test_label_batch,
                                                        self.training_flag: False})
            coarse_num+=1
            if (test_acc0 == 1):
                coarse_crrect+=1
            #大分类
            if(np.argmax(exit0_pred)==0):
                fine1_num += 1
                if(action_coarse>=action_fine_1):
                    exit1_pred, test_acc1 = sess.run([pred1, train_acc1],
                                                     feed_dict={convertPosition[action[1]]: fine_1_point,
                                                                self.label_placeholder: test_label_batch,
                                                                self.training_flag: False})
                else:
                    exit1_pred, test_acc1 = sess.run([pred1, train_acc1],
                                                     feed_dict={convertPosition[action[0]]: coarse_point,
                                                                self.label_placeholder: test_label_batch,
                                                                self.training_flag: False})
                if (test_acc1 == 1):
                    fine1_crrect += 1
            elif(np.argmax(exit0_pred)==1):
                fine2_num += 1
                if(action_coarse>=action_fine_1):
                    exit2_pred, test_acc2 = sess.run([pred2, train_acc2],
                                                     feed_dict={convertPosition[action[2]]: fine_2_point,
                                                                self.label_placeholder: test_label_batch,
                                                                self.training_flag: False})
                else:
                    exit2_pred, test_acc2 = sess.run([pred2, train_acc2],
                                                     feed_dict={convertPosition[action[0]]: coarse_point,
                                                                self.label_placeholder: test_label_batch,
                                                                self.training_flag: False})
                if (test_acc2 == 1):
                    fine2_crrect += 1
            elif(np.argmax(exit0_pred)==2):
                fine3_num += 1
                if(action_coarse>=action_fine_1):
                    exit3_pred, test_acc3 = sess.run([pred3, train_acc3],
                                                     feed_dict={convertPosition[action[3]]: fine_3_point,
                                                                self.label_placeholder: test_label_batch,
                                                                self.training_flag: False})
                else:
                    exit3_pred, test_acc3 = sess.run([pred3, train_acc3],
                                                     feed_dict={convertPosition[action[0]]: coarse_point,
                                                                self.label_placeholder: test_label_batch,
                                                                self.training_flag: False})
                if (test_acc3 == 1):
                    fine3_crrect += 1

        print('Accuracy for coarse, fine1, fine2, fine3: {}% | {}% | {}% | {}%'.format(coarse_crrect/coarse_num,
                                                                                       fine1_crrect / fine1_num,
                                                                                       fine2_crrect / fine2_num,
                                                                                       fine3_crrect / fine3_num))
        print('Overall accuracy: {} )'.format(sum([fine1_crrect, fine2_crrect, fine3_crrect]) / len(val_label)))
        sess.close()
        #overall accuracy
        return sum([fine1_crrect,fine2_crrect,fine3_crrect]) / len(val_label)

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

#     solver = Solver_Train()
#     if args.phase == 'train':
#         solver.train()
#         print("END train")
#     elif args.phase == 'test':
#         solver.test()
#         print("END test")
    solver = Solver_Train()
    solver.train()

if __name__ == '__main__':
    main()
