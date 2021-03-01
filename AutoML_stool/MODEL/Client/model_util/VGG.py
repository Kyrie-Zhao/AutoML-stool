from MODEL.Client.model_util.layers import *
import tensorflow as tf
import numpy as np

mu = 0
sigma = 0.1

class B_VGGNet(object):

    def __init__(self, num_class=10, position=[0, 0, 0, 0]):
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
            self.conv1 = Conv2d(x, filters=64, k_size=3, stride=1,name='conv1')
            self.conv1 = BN(self.conv1, phase_train=is_train,name='conv1_bn')
            self.conv1 = Relu(self.conv1,name='conv1_relu')
            self.conv2 = Conv2d(self.conv1, filters=64, k_size=3, stride=1,name='conv2')
            self.conv2 = BN(self.conv2, phase_train=is_train,name='conv2_bn')
            self.conv2 = Relu(self.conv2,name='conv2_relu')
            self.max_pool1 = max_pooling(self.conv2, k_size=2, stride=2,name='block1_maxpool')

            self.conv3 = Conv2d(self.max_pool1, filters=128, k_size=3, stride=1,name='conv3')
            self.conv3 = BN(self.conv3, phase_train=is_train,name='conv3_bn')
            self.conv3 = Relu(self.conv3,name='conv3_relu')
            self.conv4 = Conv2d(self.conv3, filters=128, k_size=3, stride=1,name='conv4')
            self.conv4 = BN(self.conv4, phase_train=is_train,name='conv4_bn')
            self.conv4 = Relu(self.conv4,name='conv4_relu')
            self.max_pool2 = max_pooling(self.conv4, k_size=2, stride=2,name='block2_maxpool')
            #-----------------------------------EXIT
            self.conv5 = Conv2d(self.max_pool2, filters=256, k_size=3, stride=1,name='conv5')
            self.conv5 = BN(self.conv5, phase_train=is_train,name='conv5_bn')
            self.conv5 = Relu(self.conv5,name='conv5_relu')
            self.conv6 = Conv2d(self.conv5, filters=256, k_size=3, stride=1,name='conv6')
            self.conv6 = BN(self.conv6, phase_train=is_train,name='conv6_bn')
            self.conv6 = Relu(self.conv6,name='conv6_relu')
            self.conv7 = Conv2d(self.conv6, filters=256, k_size=3, stride=1,name='conv7')
            self.conv7 = BN(self.conv7, phase_train=is_train,name='conv7_bn')
            self.conv7 = Relu(self.conv7,name='conv7_relu')
            self.max_pool3 = max_pooling(self.conv7, k_size=2, stride=2,name='block3_maxpool')
            # ----------------------------------EXIT
            self.conv8 = Conv2d(self.max_pool3, filters=512, k_size=3, stride=1,name='conv8')
            self.conv8 = BN(self.conv8, phase_train=is_train,name='conv8_bn')
            self.conv8 = Relu(self.conv8,name='conv8_relu')
            self.conv9 = Conv2d(self.conv8, filters=512, k_size=3, stride=1,name='conv9')
            self.conv9 = BN(self.conv9, phase_train=is_train,name='conv9_bn')
            self.conv9 = Relu(self.conv9,name='conv9_relu')
            self.conv10 = Conv2d(self.conv9, filters=512, k_size=3, stride=1,name='conv10')
            self.conv10 = BN(self.conv10, phase_train=is_train,name='conv10_bn')
            self.conv10 = Relu(self.conv10,name='conv10_relu')
            self.max_pool4 = max_pooling(self.conv10, k_size=2, stride=2,name='block4_maxpool')
            #-----------------------------------EXIT
            self.conv11 = Conv2d(self.max_pool4, filters=512, k_size=3, stride=1,name='conv11')
            self.conv11 = BN(self.conv11, phase_train=is_train,name='conv11_bn')
            self.conv11 = Relu(self.conv11,name='conv11_relu')
            self.conv12 = Conv2d(self.conv11, filters=512, k_size=3, stride=1,name='conv12')
            self.conv12 = BN(self.conv12, phase_train=is_train,name='conv12_bn')
            self.conv12 = Relu(self.conv12,name='conv12_relu')
            self.conv13 = Conv2d(self.conv12, filters=512, k_size=3, stride=1,name='conv13')
            self.conv13 = BN(self.conv13, phase_train=is_train,name='conv13_bn')
            self.conv13 = Relu(self.conv13,name='conv13_relu')

            self.fc1 = Flatten(self.conv13)
            self.fc1 = fc_layer(self.fc1, 4096,name='fc3')
            self.fc1 = Relu(self.fc1,name='fc3_relu')
            self.fc1 = Drop_out(self.fc1, 0.2, training=is_train)
            self.fc2 = fc_layer(self.fc1, 4096,name='fc4')
            self.fc2 = Relu(self.fc2,name='fc4_relu')
            self.fc2 = Drop_out(self.fc2, 0.2, training=is_train)
            logits_exit = fc_layer(self.fc2, self.num_class,name='logits_exit')

        with tf.variable_scope("coarse"):
            self.coarse = max_pooling(self.convertPosition[self.position[0]], k_size=2, stride=2,name='maxpool1')
            self.coarse = Flatten(self.coarse)
            self.coarse = fc_layer(self.coarse, 4096,name='fc1')
            self.coarse = Relu(self.coarse,name='fc1_relu')
            self.coarse = Drop_out(self.coarse, 0.2, training=is_train)
            logits_coarse = fc_layer(self.coarse, self.num_class,name='logits_coarse')

        with tf.variable_scope("fine_1"):
            self.fine_1 = max_pooling(self.convertPosition[self.position[1]], k_size=2, stride=2,name='maxpool1')
            self.fine_1 = Flatten(self.fine_1)
            self.fine_1 = fc_layer(self.fine_1, 4096,name='fc1')
            self.fine_1 = Relu(self.fine_1,name='fc1_relu')
            self.fine_1 = Drop_out(self.fine_1, 0.2, training=is_train)
            logits_fine_1= fc_layer(self.fine_1, self.num_class,name='logits_fine_1')

        with tf.variable_scope("fine_2"):
            self.fine_2 = max_pooling(self.convertPosition[self.position[2]], k_size=2, stride=2,name='maxpool1')
            self.fine_2 = Flatten(self.fine_2)
            self.fine_2 = fc_layer(self.fine_2, 4096,name='fc1')
            self.fine_2 = Relu(self.fine_2,name='fc1_relu')
            self.fine_2 = Drop_out(self.fine_2, 0.2, training=is_train)
            logits_fine_2 = fc_layer(self.fine_2, self.num_class,name='logits_fine_2')

        with tf.variable_scope("fine_3"):
            self.fine_3 = max_pooling(self.convertPosition[self.position[3]], k_size=2, stride=2,name='maxpool1')
            self.fine_3 = Flatten(self.fine_3)
            self.fine_3 = fc_layer(self.fine_3, 4096,name='fc1')
            self.fine_3 = Relu(self.fine_3,name='fc1_relu')
            self.fine_3 = Drop_out(self.fine_3, 0.2, training=is_train)
            logits_fine_2 = fc_layer(self.fine_3, self.num_class,name='logits_fine_3')
        return [logits_coarse, logits_fine_1, logits_fine_2, logits_fine_3]
