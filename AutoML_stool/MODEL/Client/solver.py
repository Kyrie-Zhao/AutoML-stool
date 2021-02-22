import os
import logging
import psutil
import time
import sys
import numpy as np
import tensorflow.compat.v1 as tf
from VGG import B_VGGNet
tf.disable_v2_behavior()

from model_util.solver_train import Solver_Train


tf.set_random_seed(123)
#np.random.seed(123)

# from AlexNet import B_VGGNet

# 一些参数用来使GPU可以使用
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
os.environ['CUDA_VISIBLE_DEVICES']='0'

class Solver(object):
    """
    这个class主要干的事：
    1： 直接运行，作为客户端的主程序
    """
    def __init__(self):

        while True:
            # 在与server交互的时候（ecrt）
            self.one_step()


    def conv_flops(self, k_size, c_in, c_out, h_out, w_out):
        return 2 * (k_size ** 2) * c_in * h_out * w_out * c_out

    def fc_flops(self,num_in, num_out):
        return 2 * num_in * num_out

    def flops_CAL(self,action):
        img_placeholder = tf.placeholder(dtype=tf.float32, shape=[1, 32, 32, 3], name='image_placeholder')
        training_flag = tf.placeholder(dtype=tf.bool, shape=[], name='training_flag')
        # create model and build graph
        B_VGGNet_instance = B_VGGNet(num_class=10,position=action)
        [logits_exit0, logits_exit1, logits_exit2, logits_exit3] = B_VGGNet_instance.model(img_placeholder, is_train=training_flag)

        layers = {}
        baseline_coarse = []
        baseline_fine_1 = []
        baseline_fine_2 = []
        baseline_fine_3 = []
        parts = {
            'baseline_1': [['baseline/conv1', 'baseline/conv2','baseline/conv3', 'baseline/conv4'], 0],
            'baseline_2': [['baseline/conv5','baseline/conv6','baseline/conv7'], 0],
            'baseline_3': [['baseline/conv8','baseline/conv9','baseline/conv10'], 0],
            'baseline_4': [['baseline/conv11', 'baseline/conv12', 'baseline/conv13', 'baseline/fc3', 'baseline/fc4', 'baseline/logits_exit3'], 0],
            'coarse': [['coarse/fc1', 'coarse/logits_coarse'], 0],
            'fine_1': [['fine_1/fc1', 'fine_1/logits_fine_1'], 0],
            'fine_2': [['fine_2/fc1', 'fine_2/logits_fine_2'], 0],
            'fine_3': [['fine_3/fc1', 'fine_3/logits_fine_3'], 0]
        }

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        print(tf.trainable_variables())
        for var in tf.trainable_variables():
            if 'conv' in var.name:
                w_out, h_out = sess.graph.get_tensor_by_name(var.name.replace('/kernel:0', '_relu:0')).get_shape().as_list()[1:3]
                #print(w_out, h_out)
                k_size, _, c_in, c_out = var.get_shape().as_list()
                print(k_size, c_in, c_out)
                op_name = var.name.replace('/kernel:0', '')
                layers[op_name] = conv_flops(k_size, c_in, c_out, h_out, w_out)
                #print(op_name, layers[op_name])

            if 'fc' in var.name or 'logits_exit' in var.name:
                print(var.name)

                num_in, num_out = sess.graph.get_tensor_by_name(var.name).get_shape().as_list()
                print(num_in)
                print(num_out)
                print("fuck")
                op_name = var.name.replace('/kernel:0', '')
                layers[op_name] = fc_flops(num_in, num_out)

        tmp = []
        for key in parts.keys():
            for layer in parts[key][0]:
                parts[key][1] += layers[layer]
            print("{}: {}".format(key, str(parts[key][1] / 1000000)+" MFLOPs"))
            tmp.append(parts[key][1] / 1000000)

        sess.close()

    def one_step(self,action):
        """
        作为class的主函数
        :return:
        """
        solver_train = Solver_Train(action)
        solver_train.train()
        #exit0_crrect / exit0_num, exit1_crrect / exit1_num, exit2_crrect / exit2_num
        accuracy = solver_train.test()
        fps = flops_CAL(action)
        print("One Step Stop")
        return [accuracy,fps]


def main():
    solver = Solver()


if __name__ == '__main__':
    main()
