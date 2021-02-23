import os
import time
import numpy as np
import argparse
import random
import string
import datetime
import tensorflow.compat.v1 as tf
import cv2
tf.disable_v2_behavior()
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

from VGG import B_VGGNet
from loss import *
from utils import read_all_batches, read_val_data, write_pickle,fileWriter
from misc import progress_bar
from augementation import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#from AlexNet import B_AlexNet
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#tf.device('/gpu:0')

class Solver_Train(object):

    def __init__(self,position):
        self.data_root = './data/cifar-10-python/cifar-10-batches-py'
        self.num_class = 10
        self.input_w = 32
        self.input_h = 32
        self.input_c = 3
        # Training parameter
        self.position = position
        self.train_batch_size = 250
        self.test_batch_size = 1
        self.lr = 0.001
        self.momentum = 0.9
        self.epochs = 40
        self.baseline_epoch = 30
        self.train_step = None
        self.test_step = None
        self.checkpoint_path = "./model_checkpoints"
        self.auto_earlyexit_lossweights = [0.03810899993000662, 0.017216535550954308, 0.05113813155702535, 0.022274152735761877]
        print(self.auto_earlyexit_lossweights)
        # Dataloader (if implemented)

        self.train_branch_acc = {}
        self.test_branch_acc = {}
        self.inference_time = {}

        self.earlyexit_lossweights = [1.0, 0.0, 0.0, 0.0]
        self.earlyexit_thresholds = [0,0,10]
        for i in range(len(self.earlyexit_lossweights)):
            self.inference_time['exit%d'%(i)] = []
            self.train_branch_acc['exit%d'%(i)] = []
            self.test_branch_acc['exit%d'%(i)] = []


    def load_data(self):
        train_data, train_label = read_all_batches(self.data_root, 5, [self.input_w, self.input_h, self.input_c])
        self.train_step = len(train_data) // self.train_batch_size
        val_data, val_label = read_val_data(self.data_root, [self.input_w, self.input_h, self.input_c], shuffle=False)
        self.test_step = len(val_data) // self.test_batch_size
        return (train_data, train_label), (val_data, val_label)


    def train(self):
        # create placeholder
        self.img_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.train_batch_size, self.input_w, self.input_h, self.input_c], name='image_placeholder')
        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[self.train_batch_size], name='label_placeholder')
        self.training_flag = tf.placeholder(dtype=tf.bool, shape=[], name='training_flag')
        self.earlyexit_lossweights_placeholder = tf.placeholder(dtype=tf.float32, shape=[len(self.earlyexit_lossweights)], name='earlyexit_lossweights_placeholder')
        #self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])

        # create model and build graph
        self.B_VGG_instance = B_VGGNet(num_class=self.num_class,position=self.position)
        [logits_exit0, logits_exit1, logits_exit2, logits_exit3] = self.B_VGG_instance.model(self.img_placeholder, is_train=self.training_flag)

        # prediction from branches
        pred0 = tf.nn.softmax(logits_exit0, name='pred_exit0')
        pred1 = tf.nn.softmax(logits_exit1, name='pred_exit1')
        pred2 = tf.nn.softmax(logits_exit2, name='pred_exit2')
        pred3 = tf.nn.softmax(logits_exit3, name='pred_exit3')

        # logits of branches
        #print(logits_exit0.shape, logits_exit1.shape, logits_exit2.shape)
        loss_exit0 = cross_entropy(logits_exit0, self.label_placeholder)
        loss_exit1 = cross_entropy(logits_exit1, self.label_placeholder)
        loss_exit2 = cross_entropy(logits_exit2, self.label_placeholder)
        loss_exit3 = cross_entropy(logits_exit3, self.label_placeholder)
        total_loss = tf.reduce_sum(tf.multiply(self.earlyexit_lossweights_placeholder, [loss_exit0, loss_exit1, loss_exit2, loss_exit3]))

        opt_exit2 = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
        #opt = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=self.momentum)
        train_op = opt_exit2.minimize(total_loss)

        # accuracy from brach
        train_acc0 = top_k_error(pred0, self.label_placeholder, 1)
        train_acc1 = top_k_error(pred1, self.label_placeholder, 1)
        train_acc2 = top_k_error(pred2, self.label_placeholder, 1)
        train_acc3 = top_k_error(pred3, self.label_placeholder, 1)


        for var in tf.trainable_variables():
            print(var.name, var.get_shape())


        # load all data in memory
        (train_data, train_label), _ = self.load_data()

        # Initialize model and create session
        init = tf.initialize_all_variables()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        #log_device_placement=True
        sess.run(init)

        # Construct saver
        saver = tf.train.Saver()

        # training record to save in pickle
        training_record = {
            'baseline_acc_record': [],
            'baseline_loss_record': [],
            'branch_acc_record': [],
            'branch_loss_record': []
        }

        print(' Start Training '.center(50, '-'))

        for epoch in range(1, self.epochs + 1):

            step_list = []
            train_error_list = []
            total_loss_list = []
            branch_loss_list = []
            val_error_list = []

            if epoch == (self.baseline_epoch+1):

                self.earlyexit_lossweights = self.auto_earlyexit_lossweights
                print('\n'+' Baseline trained '.center(50, '-'))
                print('Earlyexit loss weights update: {} '.format(self.earlyexit_lossweights))
                print('-'.center(50, '-'))
            print("\n===> epoch: %d/%d" %(epoch, self.epochs))

            for train_step_num in range(1, self.train_step+1):
                # load a batch of train data
                train_data_batch, train_label_batch = self._get_augment_train_batch(train_data, train_label, self.train_batch_size)
                if epoch > self.baseline_epoch:
                    _, train_loss, exit0_loss, exit1_loss, exit2_loss, exit3_loss, train_error0, train_error1, train_error2,train_error3 = sess.run([train_op, total_loss, loss_exit0, loss_exit1, loss_exit2, loss_exit3, train_acc0, train_acc1, train_acc2, train_acc3],
                                                                                                                            feed_dict={
                                                                                                                                self.img_placeholder: train_data_batch,
                                                                                                                                self.label_placeholder: train_label_batch,
                                                                                                                                self.earlyexit_lossweights_placeholder: self.earlyexit_lossweights,
                                                                                                                                self.training_flag: True
                                                                                                                            })
                    train_error_list.append([train_error0, train_error1, train_error2, train_error3])
                    branch_loss_list.append([exit0_loss, exit1_loss, exit2_loss, exit3_loss])
                    total_loss_list.append(train_loss)
                    format_msg = 'Loss: {:.4f} | Top1 Acc: {:.4f} | {:.4f} | {:.4f} |{:.4f} '.format(train_loss, train_error0, train_error1, train_error2, train_error3)
                    progress_bar(train_step_num, self.train_step, format_msg)

                else:
                    _, train_loss, train_error0, train_error1, train_error2, train_error3 = sess.run([train_op, total_loss, train_acc0, train_acc1, train_acc2, train_acc3],
                                                                                        feed_dict={
                                                                                            self.img_placeholder: train_data_batch,
                                                                                            self.label_placeholder: train_label_batch,
                                                                                            self.earlyexit_lossweights_placeholder: self.earlyexit_lossweights,
                                                                                            self.training_flag: True
                                                                                        })
                    train_error_list.append([train_error0, train_error1, train_error2, train_error3])
                    total_loss_list.append(train_loss)
                    format_msg = 'Loss: {:.4f} | Top1 Acc: {:.4f} | {:.4f} | {:.4f}| {:.4f}'.format(train_loss, train_error0, train_error1, train_error2, train_error3)
                    progress_bar(train_step_num, self.train_step, format_msg)

            train_error_array = np.array(train_error_list)
            print('Average loss: {:.4f} | average train epoch accuracy: {:.4f} | {:.4f} | {:.4f}| {:.4f}'.format(sum(total_loss_list) / len(total_loss_list),
                                                                                                        np.mean(train_error_array[:, 0]),
                                                                                                        np.mean(train_error_array[:, 1]),
                                                                                                        np.mean(train_error_array[:, 2]),
                                                                                                        np.mean(train_error_array[:, 3])))
            print("TRAIN END")

        save_path = saver.save(sess, os.path.join(self.checkpoint_path, 'B_VGG.ckpt'))


        sess.close()



    def test(self,action):

        # create placeholder
        self.img_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.test_batch_size, self.input_w, self.input_h, self.input_c], name='image_placeholder')
        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[self.test_batch_size], name='label_placeholder')
        self.training_flag = tf.placeholder(dtype=tf.bool, shape=[], name='training_flag')
        self.earlyexit_lossweights_placeholder = tf.placeholder(dtype=tf.float32, shape=[len(self.earlyexit_lossweights)], name='earlyexit_lossweights_placeholder')
        #self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])

        # create model and build graph
        self.B_VGG_instance = B_VGGNet(num_class=self.num_class)
        [logits_exit0, logits_exit1, logits_exit2, logits_exit3] = self.B_VGG_instance.model(self.img_placeholder, is_train=self.training_flag)


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
        _, (val_data, val_label) = self.load_data()
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
        convertPosition = [self.B_VGG_instance.conv1,self.B_VGG_instance.conv2,self.B_VGG_instance.max_pool1,self.B_VGG_instance.conv3,self.B_VGG_instance.conv4,self.B_VGG_instance.max_pool2,self.B_VGG_instance.conv5,self.B_VGG_instance.conv6,self.B_VGG_instance.conv7,self.B_VGG_instance.max_pool3,self.B_VGG_instance.conv8,self.B_VGG_instance.conv9,self.B_VGG_instance.conv10,self.B_VGG_instance.max_pool4,self.B_VGG_instance.conv11,self.B_VGG_instance.conv12,self.B_VGG_instance.conv13]
        action_coarse = action[0]
        action_fine_1 = action[1]
        action_fine_2 = action[2]
        action_fine_3 = action[3]

        for test_step_num in range(self.test_step):
            #get dataset
            test_data_batch, test_label_batch = self._get_val_batch(val_data, val_label, self.test_batch_size)
            coarse_point, fine_1_point,fine_2_point,fine_3_point,exit0_pred, test_acc0 = sess.run([convertPosition[action[0]], convertPosition[action[1]]，convertPosition[action[2]]，convertPosition[action[3]]，pred0, train_acc0], feed_dict={self.img_placeholder: test_data_batch,self.label_placeholder: test_label_batch,self.training_flag: False})
            coarse_num+=1
            if (test_acc0 == 1):
                coarse_crrect+=1
            #大分类
            if(np.argmax(exit0_pred)==0):
                fine1_num += 1
                if(action_coarse>=action_fine_1):
                    exit1_pred, test_acc1 = sess.run([pred1, train_acc1], feed_dict={convertPosition[action[1]]: fine_1_point,self.label_placeholder: test_label_batch,self.training_flag: False})
                else:
                    exit1_pred, test_acc1 = sess.run([pred1, train_acc1], feed_dict={convertPosition[action[0]]: coarse_point,self.label_placeholder: test_label_batch,self.training_flag: False})
                if (test_acc1 == 1):
                    fine1_crrect += 1
            elif(np.argmax(exit0_pred)==1):
                fine2_num += 1
                if(action_coarse>=action_fine_1):
                    exit2_pred, test_acc2 = sess.run([pred2, train_acc2], feed_dict={convertPosition[action[2]]: fine_2_point,self.label_placeholder: test_label_batch,self.training_flag: False})
                else:
                    exit2_pred, test_acc2 = sess.run([pred2, train_acc2], feed_dict={convertPosition[action[0]]: coarse_point,self.label_placeholder: test_label_batch,self.training_flag: False})
                if (test_acc2 == 1):
                    fine2_crrect += 1
            elif(np.argmax(exit0_pred)==2):
                fine3_num += 1
                if(action_coarse>=action_fine_1):
                    exit3_pred, test_acc3 = sess.run([pred3, train_acc3], feed_dict={convertPosition[action[3]]: fine_3_point,self.label_placeholder: test_label_batch,self.training_flag: False})
                else:
                    exit3_pred, test_acc3 = sess.run([pred3, train_acc3], feed_dict={convertPosition[action[0]]: coarse_point,self.label_placeholder: test_label_batch,self.training_flag: False})
                if (test_acc3 == 1):
                    fine3_crrect += 1

        print('Accuracy for coarse, fine1, fine2, fine3: {}% | {}% | {}% | {}%'.format(coarse_crrect/coarse_num, fine1_crrect / fine1_num, fine2_crrect / fine2_num, fine3_crrect / fine3_num))
        print('Overall accuracy: {} )'.format(sum([fine1_crrect,fine2_crrect,fine3_crrect]) / len(val_label)))

        sess.close()
        #overall accuracy
        return sum([fine1_crrect,fine2_crrect,fine3_crrect]) / len(val_label)

    def _entropy_cal(self, x):
        ''' Function to calculate entropy as exit condition
        '''
        #print(x)
        return np.abs(np.sum(x * np.log(x+1e-10)))

    '''
    def run(self):
        self.train()
    '''


    def _get_val_batch(self, vali_data, vali_label, vali_batch_size):
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


    def _get_augment_train_batch(self, train_data, train_labels, train_batch_size):
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


def main():


    parser = argparse.ArgumentParser(description='Branchy_AlexNet with CIFAR-10')
    # Training parameters
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset used')
    parser.add_argument('--phase', default='train', type=str, help='Train model or test')
    parser.add_argument('--epochs', default=60, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--baseline_epoch', default=40, type=int, metavar='N', help='number of epochs to train baseline')
    parser.add_argument('--trainBatchSize', default=250, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=1, type=int, help='testing batch size')
    parser.add_argument('--learning_rate', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--checkpoint_path', default='./model_checkpoints', type=str, help='Dataset used')

    # Network or loss parameters
    parser.add_argument('--auto_earlyexit_lossweights', type=float, nargs='*', dest='auto_earlyexit_lossweights',
                    default=[0.03810899993000662, 0.017216535550954308, 0.05113813155702535, 0.022274152735761877], help='List of loss weights for early exits (e.g. --lossweights 0.1 0.3)')
    parser.add_argument('--earlyexit_lossweights', type=float, nargs='*', dest='earlyexit_lossweights',
                    default=[0.0, 0.0, 0.0, 1.0], help='List of loss weights for early exits (e.g. --lossweights 0.1 0.3)')
    parser.add_argument('--earlyexit_thresholds', type=float, nargs='*', dest='earlyexit_thresholds',
                    default=[0,0,10], help='List of EarlyExit thresholds (e.g. --earlyexit 1.2 0.9)')
    #parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    args = parser.parse_args()

    solver = Solver_Train(args)

    if args.phase == 'train':
        solver.train()
        print("END train")
        #val = os.system('python solver.py --phase test &')
    elif args.phase == 'test':
        solver.test()
        #solver.test()
        print("END test")
        a = random.uniform(0.01,0.3);
        b = random.uniform(0.01,0.3);
        c = random.uniform(0.01,0.3);
        d = random.uniform(0,0.5);
        #fileWriter(str(a)+" " +str(b)+" "+str(c)+" "+str(d) + " " + str(datetime.datetime.now())+"\n",1)
        #val = os.system('python solver.py --auto_earlyexit_lossweights {} {} {} {} &'.format(a,b,c,d))
        #val = os.system("python auto.py &")
        print(str(a)+" " +str(b)+" "+str(c)+" "+str(d) )
        #print("Exit {} time_average: {} acc: {}".format(i,time_average,acc))
        #solver.test_June3()




if __name__ == '__main__':
    main()
