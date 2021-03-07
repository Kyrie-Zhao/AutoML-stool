import os
import logging
import psutil
import time
import sys
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from MODEL.Client.model_util.solver_train import Solver_Train


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

        a = 1

    def one_step(self,action):
        """
        作为class的主函数
        :return:
        """
        solver_train = Solver_Train(action)
        accuracy, flops = solver_train.train()
        #exit0_crrect / exit0_num, exit1_crrect / exit1_num, exit2_crrect / exit2_num

        print("One Step Stop")
        print(accuracy)
        print(flops)
        return [accuracy,flops]


def main():
    solver = Solver()


if __name__ == '__main__':
    main()
