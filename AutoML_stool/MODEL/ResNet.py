import tensorflow as tf
import numpy
import utils
from layers import *

class B_ResNet(object):
    def __init__(self, num_class=100, position=[5,7,9,7]):
        self.num_class = num_class
        self.position = position

        self.conv1 = None
        self.conv2_1 = None
        self.conv2_2 = None
        self.conv2_3 = None
        self.conv3_1 = None
        self.conv3_2 = None
        self.conv3_3 = None
        self.conv3_4 = None
        self.conv4_1 = None
        self.conv4_2 = None
        self.conv4_3 = None
        self.conv4_4 = None
        self.conv4_5 = None
        self.conv4_6 = None
        self.conv5_1 = None
        self.conv5_2 = None
        self.conv5_3 = None
        self.convertPosition = [self.conv1, self.conv2_1, self.conv2_2,
                                self.conv2_3, self.conv3_1, self.conv3_2,
                                self.conv3_3, self.conv3_4, self.conv4_1, self.conv4_2,
                                self.conv4_3, self.conv4_4, self.conv4_5, self.conv4_6,
                                self.conv5_1, self.conv5_2, self.conv5_3]


    def model(self, x, is_train):
        # conv layer 1
        with tf.variable_scope("0"):
            self.convertPosition[0] = conv2d(input=x, output_dim=64, kernel_size=[3, 3], stride=1, name="conv1")
            self.self.convertPosition[0] = BN(self.self.convertPosition[0], phase_train= is_train)
            self.self.convertPosition[0] = Relu(self.self.convertPosition[0],name="conv1_relu")
        with tf.variable_scope("1"):
            self.max_pooling_0_identity = conv2d(input=self.self.convertPosition[0],
                                      output_dim=256,
                                      kernel_size=[1, 1],
                                      stride=1,
                                      name="max_pooling_0_identity")
            self.max_pooling_0_identity = BN(self.max_pooling_0_identity, phase_train= is_train)
            self.convertPosition[1] = res_block(input=self.self.convertPosition[0], input_identity = self.max_pooling_0_identity, output_dim=[64,64,256], kernel_size=[(1,1),(3,3),(1,1)], end = False, name="conv2_1", phase_train=is_train)
        with tf.variable_scope("2"):
            self.convertPosition[2] = res_block(input=self.convertPosition[1], input_identity = self.convertPosition[1], output_dim=[64,64,256], kernel_size=[(1,1),(3,3),(1,1)], end = False, name="conv2_2", phase_train=is_train)
        with tf.variable_scope("3"):
            self.convertPosition[3] = res_block(input=self.convertPosition[2], input_identity = self.convertPosition[2], output_dim=[64,64,256], kernel_size=[(1,1),(3,3),(1,1)], end = False, name="conv2_3", phase_train=is_train)
        with tf.variable_scope("4"):    #identity 2
            self.con2_3_identity = conv2d(input=self.convertPosition[3],
                                      output_dim=512,
                                      kernel_size=[1,1],
                                      stride=2,
                                      name="con2_3_identity")
            self.con2_3_identity = BN(self.con2_3_identity, phase_train= is_train)
            self.convertPosition[4] = res_block(input=self.convertPosition[3], input_identity = self.con2_3_identity, output_dim=[128,128,512], kernel_size=[(1,1),(3,3),(1,1)], end = True, name = "conv3_1", phase_train=is_train)
        with tf.variable_scope("5"):
            self.convertPosition[5] = res_block(input=self.convertPosition[4], input_identity = self.convertPosition[4], output_dim=[128,128,512], kernel_size=[(1,1),(3,3),(1,1)], end = False, name="conv3_2", phase_train=is_train)
        with tf.variable_scope("6"):
            self.convertPosition[6] = res_block(input=self.convertPosition[5], input_identity = self.convertPosition[5], output_dim=[128,128,512], kernel_size=[(1,1),(3,3),(1,1)], end = False, name="conv3_3", phase_train=is_train)
        with tf.variable_scope("7"):
            self.convertPosition[7] = res_block(input=self.convertPosition[6], input_identity = self.convertPosition[6], output_dim=[128,128,512], kernel_size=[(1,1),(3,3),(1,1)], end = False, name="conv3_4", phase_train=is_train)
        with tf.variable_scope("8"):
            self.con3_4_identity = conv2d(input=self.convertPosition[7],
                                      output_dim=1024,
                                      kernel_size=[1,1],
                                      stride=2,
                                      name="con3_4_identity")
            self.con3_3_identity = BN(self.con3_4_identity, phase_train= is_train)
            self.convertPosition[8] = res_block(input=self.convertPosition[7], input_identity = self.con3_3_identity, output_dim=[256,256,1024], kernel_size=[(1,1),(3,3),(1,1)], end = True, name="conv4_1", phase_train=is_train)
        with tf.variable_scope("9"):
            self.convertPosition[9] = res_block(input=self.convertPosition[8], input_identity = self.convertPosition[8], output_dim=[256,256,1024], kernel_size=[(1,1),(3,3),(1,1)], end = False, name="conv4_2", phase_train=is_train)
        with tf.variable_scope("10"):
            self.convertPosition[10] = res_block(input=self.convertPosition[9], input_identity = self.convertPosition[9], output_dim=[256,256,1024], kernel_size=[(1,1),(3,3),(1,1)], end = False, name="conv4_3", phase_train=is_train)
        with tf.variable_scope("11"):
            self.convertPosition[11] = res_block(input=self.convertPosition[10], input_identity = self.convertPosition[10], output_dim=[256,256,1024], kernel_size=[(1,1),(3,3),(1,1)], end = False, name="conv4_4", phase_train=is_train)
        with tf.variable_scope("12"):
            self.convertPosition[12] = res_block(input=self.convertPosition[11], input_identity = self.convertPosition[11], output_dim=[256,256,1024], kernel_size=[(1,1),(3,3),(1,1)], end = False, name="conv4_5", phase_train=is_train)
        with tf.variable_scope("13"):
            self.convertPosition[13] = res_block(input=self.convertPosition[12], input_identity = self.convertPosition[12], output_dim=[256,256,1024], kernel_size=[(1,1),(3,3),(1,1)], end = False, name="conv4_6", phase_train=is_train)
        with tf.variable_scope("14"):
            self.con4_6_identity = conv2d(input=self.convertPosition[13],
                                      output_dim=2048,
                                      kernel_size=[1,1],
                                      stride=2,
                                      name="con4_6_identity")
            self.con4_6_identity = BN(self.con4_6_identity, phase_train= is_train)
            self.convertPosition[14] = res_block(input=self.convertPosition[13], input_identity = self.con4_6_identity, output_dim=[512,512,2048], kernel_size=[(1,1),(3,3),(1,1)], end = True, name="conv5_1", phase_train=is_train)
        with tf.variable_scope("15"):
            #print(self.conv5_1.shape)
            self.convertPosition[15] = res_block(input=self.convertPosition[14], input_identity = self.convertPosition[14], output_dim=[512,512,2048], kernel_size=[(1,1),(3,3),(1,1)], end = False, name="conv5_2", phase_train=is_train)
        with tf.variable_scope("16"):
            #print(self.conv5_2.shape)
            self.convertPosition[16] = res_block(input=self.convertPosition[15], input_identity = self.convertPosition[15], output_dim=[512,512,2048], kernel_size=[(1,1),(3,3),(1,1)], end = False, name="conv5_3", phase_train=is_train)

            #print(self.conv5_3.shape)
            self.avg_pooling_out = tf.layers.average_pooling2d(inputs=self.convertPosition[16], pool_size=[7,7], strides=1, padding="same")
            #print(self.avg_pooling_out.shape)
            self.avg_pooling_out = Flatten(self.conv5_3)
            #print(self.avg_pooling_out.shape)
            #print("fucksudaqiang")
            logits_exit3 = fc_layer(self.avg_pooling_out, self.num_class,name="exit3_fc")

        # exit0ault=0.1
        with tf.variable_scope("coarse"):
            self.coarse = max_pooling(self.convertPosition[self.position[0]], k_size=2, stride=2,name='maxpool1')
            self.coarse = Flatten(self.coarse)
            self.coarse = fc_layer(self.coarse, 4096,name='coarse_fc1')
            self.coarse = Relu(self.coarse,name='fc1_relu')
            self.coarse = Drop_out(self.coarse, 0.2, training=is_train)
            logits_coarse = fc_layer(self.coarse, 3, name='coarse_fc2')

        with tf.variable_scope("fine_1"):
            #print(self.position[1])
            self.fine_1 = max_pooling(self.convertPosition[self.position[1]], k_size=2, stride=2,name='maxpool1')
            self.fine_1 = Flatten(self.fine_1)
            self.fine_1 = fc_layer(self.fine_1, 4096,name='fine_1_fc1')
            self.fine_1 = Relu(self.fine_1,name='fc1_relu')
            self.fine_1 = Drop_out(self.fine_1, 0.2, training=is_train)
            logits_fine_1= fc_layer(self.fine_1, 2,name='fine_1_fc2')

        with tf.variable_scope("fine_2"):
            self.fine_2 = max_pooling(self.convertPosition[self.position[2]], k_size=2, stride=2,name='maxpool1')
            self.fine_2 = Flatten(self.fine_2)
            self.fine_2 = fc_layer(self.fine_2, 4096,name='fine_2_fc1')
            self.fine_2 = Relu(self.fine_2,name='fc1_relu')
            self.fine_2 = Drop_out(self.fine_2, 0.2, training=is_train)
            logits_fine_2 = fc_layer(self.fine_2, 3,name='fine_2_fc2')

        with tf.variable_scope("fine_3"):
            self.fine_3 = max_pooling(self.convertPosition[self.position[3]], k_size=2, stride=2,name='maxpool1')
            self.fine_3 = Flatten(self.fine_3)
            self.fine_3 = fc_layer(self.fine_3, 4096,name='fine_3_fc1')
            self.fine_3 = Relu(self.fine_3,name='fc1_relu')
            self.fine_3 = Drop_out(self.fine_3, 0.2, training=is_train)
            logits_fine_3 = fc_layer(self.fine_3, 2, name='fine_3_fc2')
        return [logits_coarse, logits_fine_1, logits_fine_2, logits_fine_3]



if __name__ == '__main__':

    g = tf.Graph()

    with g.as_default():

        input = tf.placeholder(dtype=tf.float32, shape=[1, 32, 32, 3], name="input_placeholder")

        B_wideRes_Cifar100(k=10, n=6).model(inputs=input, phase_train=tf.placeholder(tf.bool,shape=None))

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)


            # load data
            # saver = tf.train.Saver()
            # saver.restore(sess=sess, save_path="./saved/model/model")

            # show vars
            # utils.print_global_vars()
            # utils.print_trainable_vars()

            # a = g.get_tensor_by_name('res_block_3_time_1/conv2d_1/kernel:0')
            # print(a.eval())

            # save graph
            # writer = tf.summary.FileWriter("./saved/log/", sess.graph)
            # writer.close()

            # save data
            # saver = tf.train.Saver()
            # saver.save(sess=sess, save_path="./saved/model/model")
