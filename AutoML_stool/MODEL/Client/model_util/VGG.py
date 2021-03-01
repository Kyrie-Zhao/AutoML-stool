from MODEL.Client.model_util.layers import *
import tensorflow as tf
import numpy as np
import copy
mu = 0
sigma = 0.1

class B_VGGNet(object):

    def __init__(self, num_class=10, position=[5, 7, 9, 11]):
        self.num_class = num_class
        self.position = position
        self.conv1 = None
        self.conv2 = None
        self.max_pool1 = None
        self.conv3 = None
        self.conv4 = None
        self.max_pool2 = None
        self.conv5 = None
        self.conv6 = None
        self.conv7 = None
        self.max_pool3 = None
        self.conv8 = None
        self.conv9 = None
        self.conv10 = None
        self.max_pool4 = None
        self.conv11 = None
        self.conv12 = None
        self.conv13 = None
        self.convertPosition = [self.conv1, self.conv2, self.max_pool1,
                                self.conv3, self.conv4, self.max_pool2,
                                self.conv5, self.conv6, self.conv7, self.max_pool3,
                                self.conv8, self.conv9, self.conv10, self.max_pool4,
                                self.conv11, self.conv12, self.conv13]

    def model(self, x, is_train):
        # conv layer 1
        with tf.variable_scope("baseline"):
            self.convertPosition[0]  = Conv2d(x, filters=64, k_size=3, stride=1,name='conv1')
            self.convertPosition[0]  = BN(self.convertPosition[0], phase_train=is_train,name='conv1_bn')
            self.convertPosition[0]  = Relu(self.convertPosition[0],name='conv1_relu')
            self.convertPosition[1] = Conv2d(self.convertPosition[0], filters=64, k_size=3, stride=1,name='conv2')
            self.convertPosition[1] = BN(self.convertPosition[1], phase_train=is_train,name='conv2_bn')
            self.convertPosition[1] = Relu(self.convertPosition[1],name='conv2_relu')
            self.convertPosition[2] = max_pooling(self.convertPosition[1], k_size=2, stride=2,name='block1_maxpool')

            self.convertPosition[3] = Conv2d(self.convertPosition[2], filters=128, k_size=3, stride=1,name='conv3')
            self.convertPosition[3] = BN(self.convertPosition[3], phase_train=is_train,name='conv3_bn')
            self.convertPosition[3] = Relu(self.convertPosition[3],name='conv3_relu')
            self.convertPosition[4] = Conv2d(self.convertPosition[3], filters=128, k_size=3, stride=1,name='conv4')
            self.convertPosition[4] = BN(self.convertPosition[4], phase_train=is_train,name='conv4_bn')
            self.convertPosition[4] = Relu(self.convertPosition[4],name='conv4_relu')
            self.convertPosition[5] = max_pooling(self.convertPosition[4], k_size=2, stride=2,name='block2_maxpool')
            #-----------------------------------EXIT
            self.convertPosition[6] = Conv2d(self.convertPosition[5], filters=256, k_size=3, stride=1,name='conv5')
            self.convertPosition[6] = BN(self.convertPosition[6], phase_train=is_train,name='conv5_bn')
            self.convertPosition[6] = Relu(self.convertPosition[6],name='conv5_relu')
            self.convertPosition[7] = Conv2d(self.convertPosition[6], filters=256, k_size=3, stride=1,name='conv6')
            self.convertPosition[7] = BN(self.convertPosition[7], phase_train=is_train,name='conv6_bn')
            self.convertPosition[7] = Relu(self.convertPosition[7],name='conv6_relu')
            self.convertPosition[8] = Conv2d(self.convertPosition[7], filters=256, k_size=3, stride=1,name='conv7')
            self.convertPosition[8] = BN(self.convertPosition[8], phase_train=is_train,name='conv7_bn')
            self.convertPosition[8] = Relu(self.convertPosition[8],name='conv7_relu')
            self.convertPosition[9] = max_pooling(self.convertPosition[8], k_size=2, stride=2,name='block3_maxpool')
            # ----------------------------------EXIT
            self.convertPosition[10] = Conv2d(self.convertPosition[9], filters=512, k_size=3, stride=1,name='conv8')
            self.convertPosition[10] = BN(self.convertPosition[10], phase_train=is_train,name='conv8_bn')
            self.convertPosition[10] = Relu(self.convertPosition[10],name='conv8_relu')
            self.convertPosition[11] = Conv2d(self.convertPosition[10], filters=512, k_size=3, stride=1,name='conv9')
            self.convertPosition[11] = BN(self.convertPosition[11], phase_train=is_train,name='conv9_bn')
            self.convertPosition[11] = Relu(self.convertPosition[11],name='conv9_relu')
            self.convertPosition[12] = Conv2d(self.convertPosition[11], filters=512, k_size=3, stride=1,name='conv10')
            self.convertPosition[12] = BN(self.convertPosition[12], phase_train=is_train,name='conv10_bn')
            self.convertPosition[12] = Relu(self.convertPosition[12],name='conv10_relu')
            self.convertPosition[13] = max_pooling(self.convertPosition[12], k_size=2, stride=2,name='block4_maxpool')
            #-----------------------------------EXIT
            self.convertPosition[14] = Conv2d(self.convertPosition[13], filters=512, k_size=3, stride=1,name='conv11')
            self.convertPosition[14] = BN(self.convertPosition[14], phase_train=is_train,name='conv11_bn')
            self.convertPosition[14] = Relu(self.convertPosition[14],name='conv11_relu')
            self.convertPosition[15] = Conv2d(self.convertPosition[14], filters=512, k_size=3, stride=1,name='conv12')
            self.convertPosition[15] = BN(self.convertPosition[15], phase_train=is_train,name='conv12_bn')
            self.convertPosition[15] = Relu(self.convertPosition[15],name='conv12_relu')
            self.convertPosition[16] = Conv2d(self.convertPosition[15], filters=512, k_size=3, stride=1,name='conv13')
            self.convertPosition[16] = BN(self.convertPosition[16], phase_train=is_train,name='conv13_bn')
            self.convertPosition[16] = Relu(self.convertPosition[16],name='conv13_relu')

            self.fc1 = Flatten(self.convertPosition[16])
            self.fc1 = fc_layer(self.fc1, 4096,name='fc3')
            self.fc1 = Relu(self.fc1,name='fc3_relu')
            self.fc1 = Drop_out(self.fc1, 0.2, training=is_train)
            self.fc2 = fc_layer(self.fc1, 4096,name='fc4')
            self.fc2 = Relu(self.fc2,name='fc4_relu')
            self.fc2 = Drop_out(self.fc2, 0.2, training=is_train)
            logits_exit = fc_layer(self.fc2, self.num_class,name='logits_exit')

        with tf.variable_scope("coarse"):
            print(self.convertPosition[self.position[0]])
            self.coarse = max_pooling(self.convertPosition[self.position[0]], k_size=2, stride=2,name='maxpool1')
            self.coarse = Flatten(self.coarse)
            self.coarse = fc_layer(self.coarse, 4096,name='fc1')
            self.coarse = Relu(self.coarse,name='fc1_relu')
            self.coarse = Drop_out(self.coarse, 0.2, training=is_train)
            logits_coarse = fc_layer(self.coarse, 3, name='logits_coarse')

        with tf.variable_scope("fine_1"):
            self.fine_1 = max_pooling(self.convertPosition[self.position[1]], k_size=2, stride=2,name='maxpool1')
            self.fine_1 = Flatten(self.fine_1)
            self.fine_1 = fc_layer(self.fine_1, 4096,name='fc1')
            self.fine_1 = Relu(self.fine_1,name='fc1_relu')
            self.fine_1 = Drop_out(self.fine_1, 0.2, training=is_train)
            logits_fine_1= fc_layer(self.fine_1, 2,name='logits_fine_1')

        with tf.variable_scope("fine_2"):
            self.fine_2 = max_pooling(self.convertPosition[self.position[2]], k_size=2, stride=2,name='maxpool1')
            self.fine_2 = Flatten(self.fine_2)
            self.fine_2 = fc_layer(self.fine_2, 4096,name='fc1')
            self.fine_2 = Relu(self.fine_2,name='fc1_relu')
            self.fine_2 = Drop_out(self.fine_2, 0.2, training=is_train)
            logits_fine_2 = fc_layer(self.fine_2, 3,name='logits_fine_2')

        with tf.variable_scope("fine_3"):
            self.fine_3 = max_pooling(self.convertPosition[self.position[3]], k_size=2, stride=2,name='maxpool1')
            self.fine_3 = Flatten(self.fine_3)
            self.fine_3 = fc_layer(self.fine_3, 4096,name='fc1')
            self.fine_3 = Relu(self.fine_3,name='fc1_relu')
            self.fine_3 = Drop_out(self.fine_3, 0.2, training=is_train)
            logits_fine_3 = fc_layer(self.fine_3, 2, name='logits_fine_3')
        return [logits_coarse, logits_fine_1, logits_fine_2, logits_fine_3]
