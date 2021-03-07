import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
import tensorflow as tf
tf.reset_default_graph()
tf.compat.v1.disable_eager_execution()
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import tensorflow.contrib.eager as tfe

import time

import argparse
import random
import string
import datetime
import cv2

"""from VGG import B_VGGNet
from loss import *
from utils import read_all_batches, read_val_data, write_pickle
from misc import progress_bar
from augementation import *"""
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
        self.data_root = 'MODEL/Client/model_util/dataset_v4'
        #self.data_root = './dataset_v4'
        self.num_class = 3
        self.input_w = 32
        self.input_h = 32
        self.input_c = 3
        # Training parameter
        self.train_batch_size = 18
        self.test_batch_size = 1
        self.lr = 0.001
        self.momentum = 0.9
        self.epochs = int(1 * (MAX_SIZE/self.train_batch_size*7))
        self.baseline_epoch = 30
        self.train_step = None
        self.test_step = None
        self.checkpoint_path = "./model_checkpoints"
        self.auto_earlyexit_lossweights = [0.03810899993000662, 0.017216535550954308, 0.05113813155702535, 0.022274152735761877]
        self.samples_per_cls_coarse = [51, 2278,  394]
        self.samples_per_cls_fine1 = [8, 43]
        self.samples_per_cls_fine2 = [360, 1299, 619]
        self.samples_per_cls_fine3 = [354, 40]
        #print(self.auto_earlyexit_lossweights)

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
        if bristol in [0, 1, 5, 6]:
            ds = ds.map(self.read_image).map(self.augment).batch(self.train_batch_size // 18*3)
        else:
            ds = ds.map(self.read_image).map(self.augment).batch(self.train_batch_size // 18*2)
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


    def train_coarse(self,action):
        tf.reset_default_graph()

        # create placeholder
        self.img_placeholder = tf.placeholder(dtype=tf.float32,
                                              shape=[self.train_batch_size, self.input_w, self.input_h, self.input_c],
                                              name='image_placeholder')
        self.label_placeholder = tf.placeholder(dtype=tf.int32,
                                                shape=[self.train_batch_size],
                                                name='label_placeholder')
        action_coarse = action[0]
        img_placeholder_test = tf.placeholder(dtype=tf.float32,
                                              shape=[self.test_batch_size, self.input_w, self.input_h, self.input_c],
                                              name='image_placeholder_fine')

        label_placeholder_test = tf.placeholder(dtype=tf.int32,
                                                shape=[self.test_batch_size],
                                                name='label_placeholder')

        self.training_flag = tf.placeholder(dtype=tf.bool, shape=[], name='training_flag')

        self.B_VGG_instance = B_VGGNet(self.num_class,action)
        [logits_exit0, logits_exit1, logits_exit2, logits_exit3] = self.B_VGG_instance.model(self.img_placeholder, # 2 F1
                                                                               is_train=self.training_flag)
        print('VGG DONE')
        # prediction from branches
        pred0 = tf.nn.softmax(logits_exit0, name='pred_exit0')
        print('balanced loss')
        # logits of branches
        #balanced loss
        loss_exit0 = class_balanced_cross_entropy_loss(logits_exit0, self.label_placeholder, self.samples_per_cls_coarse)
        print('balanced loss')
        #loss_exit0 = cross_entropy(logits_exit0, self.label_placeholder)
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
        best_test_acc = 0
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
            X_train = sess.run(X_train)
            y_c_train = sess.run(y_c_train)
            _, train_loss, train_error0 = sess.run([train_op, loss_exit0, train_acc0],
                                                   feed_dict={self.img_placeholder: X_train,
                                                              self.label_placeholder: y_c_train,
                                                              self.training_flag: True})
            train_error_list.append([train_error0])
            total_loss_list.append(train_loss)
            #print('Loss: {:.4f}'.format(train_loss))

            #print('Acc: {:.4f} | {:.4f} | {:.4f}| {:.4f}'.format(train_error0, train_error1, train_error2, train_error3))
            progress_bar(epoch, self.epochs + 1)

            """if (epoch%int(MAX_SIZE/self.train_batch_size*7)==0):
                print("fuk")
                ds_test = self.load_data_test()
                batch_test = ds_test.make_one_shot_iterator().get_next()
                coarse_num = 0
                coarse_crrect = 0
                for i in range(552):
                    X_test, y_b_test, y_c_test, y_bc_test = sess.run(batch_test)
                    coarse_point,exit0_pred, test_acc0 = sess.run([self.B_VGG_instance.convertPosition[action_coarse],
                                                      pred0, train_acc0],
                                                     feed_dict={img_placeholder_test: X_test,
                                                                label_placeholder_test: y_c_test,
                                                                self.training_flag: False})
                    coarse_num+=1
                    if (test_acc0 == 1):
                        coarse_crrect+=1
                if (coarse_crrect/coarse_num>best_test_acc):
                    save_path = saver.save(sess, os.path.join(self.checkpoint_path, './coarse.ckpt'))
                    best_test_acc = coarse_test(action)
                    print("best accuracy:")
                    print(best_test_acc)"""
        save_path = saver.save(sess, os.path.join(self.checkpoint_path, './coarse.ckpt'))
        sess.close()
        print("Train coarse end")
        #exit(1)

    def train_fine(self,action):
        #position=[5,7,9,7]
        tf.reset_default_graph()

        # create placeholder
        #self.ip = tf.placeholder(dtype=tf.float32,shape=[6,7,7,128])
        img_placeholder = tf.placeholder(dtype=tf.float32,
                                              shape=[self.train_batch_size/3, self.input_w, self.input_h, self.input_c],
                                              name='image_placeholder_fine')

        label_placeholder = tf.placeholder(dtype=tf.int32,
                                                shape=[self.train_batch_size/3],
                                                name='label_placeholder')

        training_flag = tf.placeholder(dtype=tf.bool, shape=[], name='training_flag')
        """earlyexit_lossweights_placeholder = tf.placeholder(dtype=tf.float32,
                                                                shape=[3],
                                                                name='earlyexit_lossweights_placeholder')"""
        # create model and build graph
        print("build fine model")
        B_VGG_instance = B_VGGNet(self.num_class,action)
        #print("opt fine")
        [logits_exit0, logits_exit1, logits_exit2, logits_exit3] = B_VGG_instance.model(img_placeholder,
                                                                                             is_train=training_flag)
        #print("opt fine")
        #Action
        """convertPosition=[self.B_VGG_instance.conv1, self.B_VGG_instance.conv2, self.B_VGG_instance.max_pool1,
                         self.B_VGG_instance.conv3, self.B_VGG_instance.conv4, self.B_VGG_instance.max_pool2,
                         self.B_VGG_instance.conv5, self.B_VGG_instance.conv6, self.B_VGG_instance.conv7,
                         self.B_VGG_instance.max_pool3, self.B_VGG_instance.conv8, self.B_VGG_instance.conv9,
                         self.B_VGG_instance.conv10, self.B_VGG_instance.max_pool4, self.B_VGG_instance.conv11,
                         self.B_VGG_instance.conv12, self.B_VGG_instance.conv13]"""
        action_coarse = action[0]
        action_fine_1 = action[1]
        action_fine_2 = action[2]
        action_fine_3 = action[3]
        # prediction from branches
        pred1 = tf.nn.softmax(logits_exit1, name='pred_exit1')
        pred2 = tf.nn.softmax(logits_exit2, name='pred_exit2')
        pred3 = tf.nn.softmax(logits_exit3, name='pred_exit3')

        loss_exit1 = class_balanced_cross_entropy_loss(logits_exit1, label_placeholder, self.samples_per_cls_fine1)
        loss_exit2 = class_balanced_cross_entropy_loss(logits_exit2, label_placeholder, self.samples_per_cls_fine2)
        loss_exit3 = class_balanced_cross_entropy_loss(logits_exit3, label_placeholder, self.samples_per_cls_fine3)
        """
        loss_exit1 = cross_entropy(logits_exit1, label_placeholder)
        loss_exit2 = cross_entropy(logits_exit2, label_placeholder)
        loss_exit3 = cross_entropy(logits_exit3, label_placeholder)"""
        """total_loss = tf.reduce_sum(tf.multiply(earlyexit_lossweights_placeholder,
                                               [loss_exit1, loss_exit2, loss_exit3]))"""
        # logits of branches
        # accuracy from brach

        tmp_scope_fine_1 = ""
        if action_fine_1<=action_coarse:
            tmp_scope_fine_1 = 'fine_1'
        else:
            for tmp in range(action_coarse+1,action_fine_1+1):
                tmp_scope_fine_1 = tmp_scope_fine_1+str(tmp)+"|"
            tmp_scope_fine_1 = tmp_scope_fine_1+'fine_1'
        update_ops_fine_1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope=tmp_scope_fine_1)
        #print(update_ops_fine_1)

        tmp_scope_fine_2 = ""
        if action_fine_2<=action_coarse:
            tmp_scope_fine_2 = 'fine_2'
        else:
            for tmp in range(action_coarse+1,action_fine_2+1):
                tmp_scope_fine_2 = tmp_scope_fine_2+str(tmp)+"|"
            tmp_scope_fine_2 = tmp_scope_fine_2+'fine_2'
        update_ops_fine_2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope=tmp_scope_fine_2)
        #print(update_ops_fine_2)
        tmp_scope_fine_3 = ""
        if action_fine_3<=action_coarse:
            tmp_scope_fine_3 = 'fine_3'
        else:
            for tmp in range(action_coarse+1,action_fine_3+1):
                tmp_scope_fine_3 = tmp_scope_fine_3+str(tmp)+"|"
            tmp_scope_fine_3 = tmp_scope_fine_3+'fine_2'
        update_ops_fine_3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope=tmp_scope_fine_3)
        #print(update_ops_fine_3)

        opt_exit2 = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
        #print(update_ops)
        print("prepare fine models")
        train_op_1 = opt_exit2.minimize(loss_exit1, var_list=update_ops_fine_1)
        train_op_2 = opt_exit2.minimize(loss_exit2, var_list=update_ops_fine_2)
        train_op_3 = opt_exit2.minimize(loss_exit3, var_list=update_ops_fine_3)
        #train_op_loss_exit1 = opt_exit2.minimize(loss_exit1)
        #print("train fine begins1")
        train_acc1 = top_k_error(pred1, label_placeholder, 1)
        train_acc2 = top_k_error(pred2, label_placeholder, 1)
        train_acc3 = top_k_error(pred3, label_placeholder, 1)

        module_file = tf.train.latest_checkpoint(os.path.join(self.checkpoint_path, './coarse.ckpt'))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if module_file is not None:
                saver.restore(sess, module_file)
            saver = tf.train.Saver()
            """sess = tf.Session()
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(self.checkpoint_path, './coarse.ckpt'))"""
            #variables = tf.get_collection(tf.GraphKeys.VARIABLES)
            #for i in variables:
            #    print(i)
            ds_train_0, ds_train_1, ds_train_2, ds_train_3, ds_train_4, ds_train_5, ds_train_6 = self.load_data_train()
            batch_0 = ds_train_0.make_one_shot_iterator().get_next()
            batch_1 = ds_train_1.make_one_shot_iterator().get_next()
            batch_2 = ds_train_2.make_one_shot_iterator().get_next()
            batch_3 = ds_train_3.make_one_shot_iterator().get_next()
            batch_4 = ds_train_4.make_one_shot_iterator().get_next()
            batch_5 = ds_train_5.make_one_shot_iterator().get_next()
            batch_6 = ds_train_6.make_one_shot_iterator().get_next()
            print("train fine begins")
            total_loss_list = []
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

                X_train_f0 = sess.run(X_train_f0)
                y_bc_train_f0 = sess.run(y_bc_train_f0)
                #print(y_bc_train_f0)
                X_train_f1 = sess.run(X_train_f1)
                y_bc_train_f1 = sess.run(y_bc_train_f1)
                #print(y_bc_train_f1)
                X_train_f2 = sess.run(X_train_f2)
                y_bc_train_f2 = sess.run(y_bc_train_f2)
                #print(y_bc_train_f2)
                train_loss, train_error1 = sess.run([train_op_1, train_acc1],
                                                       feed_dict={img_placeholder: X_train_f0,
                                                                  label_placeholder: y_bc_train_f0,
                                                                  training_flag: True})
                train_loss, train_error1 = sess.run([train_op_2, train_acc2],
                                                       feed_dict={img_placeholder: X_train_f1,
                                                                  label_placeholder: y_bc_train_f1,
                                                                  training_flag: True})
                train_loss, train_error2 = sess.run([train_op_3, train_acc3],
                                                       feed_dict={img_placeholder: X_train_f2,
                                                                  label_placeholder: y_bc_train_f2,
                                                                  training_flag: True})
                #FINE1
                """if(action_fine_1<=action_coarse):
                    fine1_out = sess.run(self.B_VGG_instance.convertPosition[action_fine_1],
                                         feed_dict={img_placeholder: X_train_f0,
                                                    label_placeholder: y_bc_train_f0,
                                                    training_flag: False})

                    _, train_loss, train_error1 = sess.run([train_op, total_loss, train_acc1],
                                                           feed_dict={self.B_VGG_instance.convertPosition[action_fine_1]: fine1_out,
                                                                      label_placeholder: y_bc_train_f0,
                                                                      earlyexit_lossweights_placeholder: [0.3,0.4,0.3],
                                                                      training_flag: True})

                else:
                    coarse_out = sess.run(self.B_VGG_instance.convertPosition[action_coarse],
                                          feed_dict={img_placeholder: X_train_f0,
                                                     label_placeholder: y_bc_train_f0,
                                                     training_flag: False})
                    _,train_loss, train_error1 = sess.run([train_op,total_loss, train_acc1],
                                                           feed_dict={self.B_VGG_instance.convertPosition[action_coarse]: coarse_out,
                                                                      label_placeholder: y_bc_train_f0,
                                                                      earlyexit_lossweights_placeholder: [0.3,0.4,0.3],
                                                                      training_flag: True})
                    print(train_loss)
                    print(_)

                if(action_fine_2<=action_coarse):
                    fine2_out = sess.run(self.B_VGG_instance.convertPosition[action_fine_2], feed_dict={img_placeholder: X_train_f1,
                                                                                      label_placeholder: y_bc_train_f1,
                                                                                      training_flag: False})
                    _, train_loss, train_error2 = sess.run([train_op, total_loss, train_acc2],
                                                           feed_dict={self.B_VGG_instance.convertPosition[action_fine_2]: fine2_out,
                                                                      label_placeholder: y_bc_train_f1,
                                                                      earlyexit_lossweights_placeholder: [0.3,0.4,0.3],
                                                                      training_flag: True})
                else:
                    coarse_out = sess.run(self.B_VGG_instance.convertPosition[action_coarse], feed_dict={img_placeholder: X_train_f1,
                                                                                       label_placeholder: y_bc_train_f1,
                                                                                       training_flag: False})
                    _, train_loss, train_error2 = sess.run([train_op, total_loss, train_acc2],
                                                           feed_dict={self.B_VGG_instance.convertPosition[action_coarse]: coarse_out,
                                                                      label_placeholder: y_bc_train_f1,
                                                                      earlyexit_lossweights_placeholder: [0.3,0.4,0.3],
                                                                      training_flag: True})


                if(action_fine_3<=action_coarse):
                    fine3_out = sess.run(self.B_VGG_instance.convertPosition[action_fine_3], feed_dict={img_placeholder: X_train_f2,
                                                                                      label_placeholder: y_bc_train_f2,
                                                                                      self.training_flag: False})
                    _, train_loss, train_error3 = sess.run([train_op, total_loss, train_acc3],
                                                           feed_dict={self.B_VGG_instance.convertPosition[action_fine_3]: fine3_out,
                                                                      label_placeholder: y_bc_train_f2,
                                                                      earlyexit_lossweights_placeholder: [0.3,0.4,0.3],
                                                                      training_flag: True})
                else:
                    coarse_out = sess.run(self.B_VGG_instance.convertPosition[action_coarse],
                                          feed_dict={img_placeholder: X_train_f2,
                                                     label_placeholder: y_bc_train_f2,
                                                     training_flag: False})
                    _, train_loss, train_error3 = sess.run([train_op, total_loss, train_acc3],
                                                           feed_dict={self.B_VGG_instance.convertPosition[action_coarse]: coarse_out,
                                                                      label_placeholder: y_bc_train_f2,
                                                                      earlyexit_lossweights_placeholder: [0.3,0.4,0.3],
                                                                      training_flag: True})"""

                #train_error_list.append([train_error0])
                total_loss_list.append(train_loss)
                #print('Loss: {:.4f}'.format(train_loss))
                progress_bar(epoch, self.epochs + 1)

            print("train end")
            #tf.get_variable_scope().reuse_variables()
            save_path = saver.save(sess, os.path.join(self.checkpoint_path, './AutoML_stool.ckpt'))
            sess.close()

            #FINE2
            #FINE3
    def coarse_test(self,action):
        tf.reset_default_graph()
        print("coarse test begins")
        sess = tf.Session()
        img_placeholder = tf.placeholder(dtype=tf.float32,
                                              shape=[self.test_batch_size, self.input_w, self.input_h, self.input_c],
                                              name='image_placeholder')
        label_placeholder = tf.placeholder(dtype=tf.int32,
                                                shape=[self.test_batch_size],
                                                name='label_placeholder')
        training_flag = tf.placeholder(dtype=tf.bool, shape=[], name='training_flag')
        B_VGG_instance = B_VGGNet(self.num_class,action)
        [logits_exit0, logits_exit1, logits_exit2, logits_exit3] = B_VGG_instance.model(img_placeholder,is_train=training_flag)
        action_coarse = action[0]
        pred0 = tf.nn.softmax(logits_exit0, name='pred_exit0')
        train_acc0 = top_k_error(pred0, label_placeholder, 1)
        module_file = tf.train.latest_checkpoint(os.path.join(self.checkpoint_path, './coarse.ckpt'))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if module_file is not None:
                saver.restore(sess, module_file)
        ds_test = self.load_data_test()
        batch_test = ds_test.make_one_shot_iterator().get_next()
        for i in range(552):

            X_test, y_b_test, y_c_test, y_bc_test = sess.run(batch_test)
            coarse_point,exit0_pred, test_acc0 = sess.run([self.B_VGG_instance.convertPosition[action_coarse],
                                              pred0, train_acc0],
                                             feed_dict={img_placeholder: X_test,
                                                        label_placeholder: y_c_test,
                                                        training_flag: False})
            coarse_num+=1
            if (test_acc0 == 1):
                coarse_crrect+=1
        return coarse_crrect/coarse_num

    def test(self, action):
        tf.reset_default_graph()
        print("test begins")
        #action = [5,7,9,7]
        coarse_num = 0

        coarse_crrect = 0
        fine1_crrect = 0
        fine2_crrect = 0
        fine3_crrect = 0

        coarse_num = 0
        fine1_num = 0
        fine2_num = 0
        fine3_num = 0
        sess = tf.Session()
        img_placeholder = tf.placeholder(dtype=tf.float32,
                                              shape=[self.test_batch_size, self.input_w, self.input_h, self.input_c],
                                              name='image_placeholder')
        label_placeholder = tf.placeholder(dtype=tf.int32,
                                                shape=[self.test_batch_size],
                                                name='label_placeholder')
        training_flag = tf.placeholder(dtype=tf.bool, shape=[], name='training_flag')

        # create model and build graph

        self.B_VGG_instance = B_VGGNet(self.num_class,action)
        [logits_exit0, logits_exit1, logits_exit2, logits_exit3] = self.B_VGG_instance.model(img_placeholder,
                                                                                             is_train=training_flag)

        #Action

        action_coarse = action[0]
        action_fine_1 = action[1]
        action_fine_2 = action[2]
        action_fine_3 = action[3]

        # prediction from branches
        pred0 = tf.nn.softmax(logits_exit0, name='pred_exit0')
        pred1 = tf.nn.softmax(logits_exit1, name='pred_exit1')
        pred2 = tf.nn.softmax(logits_exit2, name='pred_exit2')
        pred3 = tf.nn.softmax(logits_exit3, name='pred_exit3')

        train_acc0 = top_k_error(pred0, label_placeholder, 1)
        """train_acc1 = top_k_error(pred1, label_placeholder, 1)
        train_acc2 = top_k_error(pred2, label_placeholder, 1)
        train_acc3 = top_k_error(pred3, label_placeholder, 1)"""

        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(self.checkpoint_path, './AutoML_stool.ckpt'))

        ds_test = self.load_data_test()
        batch_test = ds_test.make_one_shot_iterator().get_next()

        label_list = []
        classifier_list = []
        for i in range(552):
            if i%100==0:
                print(str(i)+" samples tested")
            X_test, y_b_test, y_c_test, y_bc_test = sess.run(batch_test)
            coarse_point, fine_1_point, fine_2_point, fine_3_point, exit0_pred, test_acc0 = sess.run([self.B_VGG_instance.convertPosition[action_coarse], self.B_VGG_instance.convertPosition[action_fine_1],self.B_VGG_instance.convertPosition[action_fine_2], self.B_VGG_instance.convertPosition[action_fine_3],
                                              pred0, train_acc0],
                                             feed_dict={img_placeholder: X_test,
                                                        label_placeholder: y_c_test,
                                                        training_flag: False})

            coarse_num+=1
            if (test_acc0 == 1):
                coarse_crrect+=1
            #大分类

            if(np.argmax(exit0_pred)==0):
                fine1_num += 1
                exit1_pred= sess.run(pred1,feed_dict={self.B_VGG_instance.convertPosition[action_fine_1]: fine_1_point,
                                                            label_placeholder: y_bc_test,
                                                            training_flag: False})
                if (test_acc0 == 1):
                    label_list.append(np.argmax(exit1_pred))
                else:
                    label_list.append(np.argmax(exit1_pred)+10)
                classifier_list.append(y_bc_test[0])
                #print(y_bc_test)
                """if(action_coarse>=action_fine_1):
                    exit1_pred, test_acc1 = sess.run([pred1, train_acc1],
                                                     feed_dict={self.B_VGG_instance.convertPosition[action_fine_1]: fine_1_point,
                                                                label_placeholder: y_bc_test,
                                                                training_flag: False})
                else:
                    exit1_pred, test_acc1 = sess.run([pred1, train_acc1],
                                                     feed_dict={self.B_VGG_instance.convertPosition[action_coarse]: coarse_point,
                                                                label_placeholder: y_bc_test,
                                                                training_flag: False})"""
                if (test_acc0 == 1) and (np.argmax(exit1_pred)==y_bc_test[0]):
                    fine1_crrect += 1
            elif(np.argmax(exit0_pred)==1):
                fine2_num += 1
                exit2_pred = sess.run(pred2,feed_dict={self.B_VGG_instance.convertPosition[action_fine_2]: fine_2_point,
                                                            label_placeholder: y_bc_test,
                                                            training_flag: False})
                if (test_acc0 == 1):
                    label_list.append(np.argmax(exit2_pred))
                else:
                    label_list.append(np.argmax(exit2_pred)+10)
                classifier_list.append(y_bc_test[0])
                """if(action_coarse>=action_fine_1):
                    exit2_pred, test_acc2 = sess.run([pred2, train_acc2],
                                                     feed_dict={self.B_VGG_instance.convertPosition[action_fine_2]: fine_2_point,
                                                                label_placeholder: y_bc_test,
                                                                training_flag: False})
                else:
                    exit2_pred, test_acc2 = sess.run([pred2, train_acc2],
                                                     feed_dict={self.B_VGG_instance.convertPosition[action_coarse]: coarse_point,
                                                                label_placeholder: y_bc_test,
                                                                training_flag: False})"""
                if (test_acc0 == 1) and (np.argmax(exit2_pred)==y_bc_test[0]):
                    fine2_crrect += 1
            elif(np.argmax(exit0_pred)==2):
                fine3_num += 1
                exit3_pred = sess.run(pred3,feed_dict={self.B_VGG_instance.convertPosition[action_fine_3]: fine_3_point,
                                                            label_placeholder: y_bc_test,
                                                            training_flag: False})
                if (test_acc0 == 1):
                    label_list.append(np.argmax(exit3_pred))
                else:
                    label_list.append(np.argmax(exit3_pred)+10)
                classifier_list.append(y_bc_test[0])
                """if(action_coarse>=action_fine_1):
                    exit3_pred, test_acc3 = sess.run([pred3, train_acc3],
                                                     feed_dict={self.B_VGG_instance.convertPosition[action_fine_3]: fine_3_point,
                                                                label_placeholder: y_bc_test,
                                                                training_flag: False})
                else:
                    exit3_pred, test_acc3 = sess.run([pred3, train_acc3],
                                                     feed_dict={self.B_VGG_instance.convertPosition[action_coarse]: coarse_point,
                                                                label_placeholder: y_bc_test,
                                                                training_flag: False})"""
                if (test_acc0 == 1) and (np.argmax(exit3_pred)==y_bc_test[0]):
                    fine3_crrect += 1


            # y_c_pred = coarses_model(X_test) #0, 1, 2
            # y_bc_fine = fine_model(X_test, y_c_pred) # 0, 1 or 0,1, 2 or 0, 1
            # if y_c_pred == 0， y_b_pred = y_bc_pred
            # == 1, +2
            # == 2， +5

        """print('Accuracy for coarse, fine1, fine2, fine3: {}% | {}% | {}% | {}%'.format(coarse_crrect/coarse_num,
                                                                                       fine1_crrect / fine1_num,
                                                                                       fine2_crrect / fine2_num,
                                                                                       fine3_crrect / fine3_num))"""
        balanced_accuracy = balanced_accuracy_score(classifier_list, label_list)
        print(len(label_list))
        print('Overall accuracy: {} )'.format(balanced_accuracy))
        #print('Fine branches number:  {} | {} | {} '.format(fine1_num,fine2_num,fine3_num))
        sess.close()
        #overall accuracy
        return [balanced_accuracy, fine1_num,fine2_num,fine3_num]

    def conv_flops(self, k_size, c_in, c_out, h_out, w_out):
        return 2 * (k_size ** 2) * c_in * h_out * w_out * c_out

    def fc_flops(self, num_in, num_out):
        return 2 * num_in * num_out

    def flops_Cal(self,fine1_num, fine2_num, fine3_num, action):
        tf.reset_default_graph()
        print("flops cal")
        img_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.test_batch_size, self.input_w, self.input_h, self.input_c], name='image_placeholder')
        training_flag = tf.placeholder(dtype=tf.bool, shape=[], name='training_flag')
        # create model and build graph
        B_VGGNet_instance = B_VGGNet(10,action)
        #Action
        convertPosition=["0/conv1", "1/conv2", "2/max_pool1",
                         "3/conv3", "4/conv4", "5/max_pool2",
                         "6/conv5", "7/conv6", "8/conv7",
                         "9/max_pool3", "10/conv8", "11/conv9",
                         "12/conv10", "13/max_pool4", "14/conv11",
                         "15/conv12", "16/conv13"]
        action_coarse = action[0]
        action_fine_1 = action[1]
        action_fine_2 = action[2]
        action_fine_3 = action[3]

        coarse_line = []
        fine_1_line = []
        fine_2_line = []
        fine_3_line = []
        tmp_line = []
        for i in convertPosition[:(action_coarse+1)]:
            if "max_pool" not in i:
                tmp_line.append(i)
        coarse_line.append(tmp_line)
        coarse_line.append(0)
        tmp_line = []
        for i in convertPosition[:(action_fine_1+1)]:
            if "max_pool" not in i:
                tmp_line.append(i)
        fine_1_line.append(tmp_line)
        fine_1_line.append(0)
        tmp_line = []
        for i in convertPosition[:(action_fine_2+1)]:
            if "max_pool" not in i:
                tmp_line.append(i)
        fine_2_line.append(tmp_line)
        fine_2_line.append(0)
        tmp_line = []
        for i in convertPosition[:(action_fine_3+1)]:
            if "max_pool" not in i:
                tmp_line.append(i)
        fine_3_line.append(tmp_line)
        fine_3_line.append(0)
        [logits_exit0, logits_exit1, logits_exit2, logits_exit3] = B_VGGNet_instance.model(img_placeholder, is_train=training_flag)

        layers = {}
        parts = {
            'coarse': coarse_line,
            'fine_1': fine_1_line,
            'fine_2': fine_2_line,
            'fine_3': fine_3_line,
            'exit0': [['coarse/coarse_fc1', 'coarse/coarse_fc2'], 0],
            'exit1': [['fine_1/fine_1_fc1', 'fine_1/fine_1_fc2'], 0],
            'exit2': [['fine_2/fine_2_fc1', 'fine_2/fine_2_fc2'], 0],
            'exit3': [['fine_3/fine_3_fc1', 'fine_3/fine_3_fc2'], 0]
            }
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        #print(tf.trainable_variables())
        #print("zzh")
        for var in tf.trainable_variables():
            if 'conv' in var.name:
                #print(var.name)
                #print("conv")
                w_out, h_out = sess.graph.get_tensor_by_name(var.name.replace('/kernel:0', '_relu:0')).get_shape().as_list()[1:3]
                k_size, _, c_in, c_out = var.get_shape().as_list()
                #print(k_size, c_in, c_out)
                op_name = var.name.replace('/kernel:0', '')
                layers[op_name] = self.conv_flops(k_size, c_in, c_out, h_out, w_out)
                #print(op_name, layers[op_name])

            if 'fc' in var.name or 'logits_exit' in var.name:
                #print(var.name)
                #print("fc")
                num_in, num_out = sess.graph.get_tensor_by_name(var.name).get_shape().as_list()
                #print(num_in)
                #print(num_out)
                op_name = var.name.replace('/kernel:0', '')
                layers[op_name] = self.fc_flops(num_in, num_out)

        tmp = []
        print("var stop1")
        for key in parts.keys():
            for layer in parts[key][0]:
                parts[key][1] += layers[layer]
                #print(layers[layer])
                #print(layer)
                #print(parts)

            #print("{}: {}".format(key, str(parts[key][1] / 1000000)+" MFLOPs"))
            tmp.append(parts[key][1] / 1000000)
            #print("append")
            #print(parts[key][1])
        #print(tmp)
        coarse_flops = tmp[0]+tmp[4]
        fine1_flops = tmp[1]+tmp[5]
        fine2_flops = tmp[2]+tmp[6]
        fine3_flops = tmp[3]+tmp[7]
        flops = (fine1_num*fine1_flops+fine2_num*fine2_flops+fine3_num*fine3_flops+552*coarse_flops)/552

        print("coarse_flops: {}, fine1_flops: {}, fine2_flops: {}, fine3_flops: {}, total flops: {} MFLOPS".format(coarse_flops,fine1_flops,fine2_flops,fine3_flops,flops))
        sess.close()
        return flops

    def train(self):
        #self.position = [5,7,9,7]
        self.train_coarse(self.position)

        #self.position = [5,7,9,7]
        self.train_fine(self.position)
        accuracy, fine1_num, fine2_num, fine3_num = self.test(self.position)
        #print("overall accuracy")
        #print(accuracy)
        flops = self.flops_Cal(fine1_num, fine2_num, fine3_num,self.position)

        return [accuracy, flops]

    def test_bk(self,action):


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
        self.B_VGG_instance = B_VGGNet(self.num_class,action)
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
