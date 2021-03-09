import os
import numpy as np
import tensorflow as tf
from MODEL.solver_train import Solver_Train

tf.set_random_seed(123)

class Solver(object):

    def __init__(self):
        a = 1

    def one_step(self,action):

        solver_train = Solver_Train(action)
        accuracy, flops = solver_train.train()
        print('Solver: Balanced Accuracy: {}%, Flops: {}%'.format(accuracy, flops))
        return [accuracy, flops]

def main():
    solver = Solver()


# if __name__ == '__main__':
#     main()
