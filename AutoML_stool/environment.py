import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

import math 
import argparse
import time
import pickle
import numpy as np
import tensorflow as tf
import json
from RL_AGENT.ddpg import Ddpg
import matplotlib.pyplot as plt
from MODEL.solver import Solver

class Environment(object):
    """
    The Main Environment to connect the trainer with the rl agent
    """
    def __init__(self, dis_dim, scal_dim, s_dim, scal_var, a_bound, s_bound, train=True):
        """
        init trainer and rl_agent
        :param train: Train rl or not
        :param solver: Solver class, which have .one_step
        :param rl_model: rl_model class, which have .choose_action, .store_transition and .learn
        """
        self.update_RL = 1
        self.flag_DA = 0
        # init param for rl
        self.dis_dim = dis_dim
        self.scal_dim = scal_dim
        self.s_dim = s_dim
        self.scal_var = scal_var
        self.a_bound = a_bound
        self.s_bound = s_bound
        self.da_memory = []
        self.train = train
        self.solver = Solver()
        self.rl_model = Ddpg(self.dis_dim, self.scal_dim, self.s_dim, self.scal_var, self.a_bound)

        if (self.flag_DA == 1):
            da_extractedData = np.load('da.npy', allow_pickle=True).tolist()
            for i in range(0,len(da_extractedData)):
                self.rl_model.store_transition(da_extractedData[i][0], da_extractedData[i][1],
                                               da_extractedData[i][2],da_extractedData[i][3])
            print("Restore Domain Adaptation Successfully!")
        self.random_step = 1
        self.global_step = 0
        self.global_epoch = 0
        self.init_param_to_store()

    def init_param_to_store(self):
        self.step = []
        self.accuracy = []
        self.action = []
        self.fps = []
        self.epoch = []
        self.accuracy_epoch = []
        self.fps_epoch = []
        self.reward = []
        # config:
        return

    def store_all(self):
        step = {
            "step": self.step,
            "accuracy": self.accuracy,
            "fps":self.fps,
            "output_point": self.output_point,
        }

        epoch = {
            "epoch": self.epoch,
            "accuracy": self.accuracy_epoch,
            "fps":self.fps_epoch,
            "reward": self.reward
        }

        all = {
            "step": step,
            "epoch": epoch
        }
        file = open("./data.pk", "wb")
        pickle.dump(all, file)
        file.close()

    def init_state(self):
        """
        Init_state if state is empty
        Using mean of s_bound and std of normal generate s_bound
        """
        # calculate mean and std according to s_bound
        print("Initialize the state")
        mean = np.multiply(self.s_bound, 0.5)
        std = np.multiply(self.s_bound, 0.1)
        # generate normal
        state = np.random.normal(mean, std)
        return state

    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    def calculate_p(self, acc, fps):
        reward = -(1-acc)*fps
        return reward
    
    def calculate_reward(self, state):
        acc, fps = state[0], state[1]
        reward = self.calculate_p(acc, fps)
        print(acc, fps, reward)
        return reward

    def take_action(self, action, late_rl):
        """
        Used function:
            solver.one_step => a function mapping action to new state
            calculate_reward => a function mapping state to reward

        """
        print('Take action:', action)
        self.global_epoch += 1

        next_state_to_average = []
        out_point_avg = []

        self.global_step += 1

        next_state = self.solver.one_step(action)
        next_state[1] = math.log10(next_state[1])
        next_state.append(action[0])
        next_state.append(action[1])
        next_state.append(action[2])
        next_state.append(action[3])
        
        self.step.append(self.global_step)
        self.accuracy.append(next_state[0])
        self.fps.append(next_state[1])

        reward = self.calculate_reward(next_state)

        self.epoch.append(self.global_epoch)
        self.accuracy_epoch.append(next_state[0])
        self.fps_epoch.append(next_state[1])
        self.reward.append(reward)

        return np.array(next_state), reward

    def main(self, epoch):
        """
        Main function of the whole program
        :param epoch: total epoch for rl to run

        How it work:
            For every learning epoch:
                Get last state s (if don't have one, init one)
                Use rl_model.take_action() to generate action pair
                Give action pair to Solver.one_step() to generate new state s_ for rl
                Then calculate reward by env.calculate_reward() get reward r
                Store the transition [s, a, [r], s] by using rl_model.save_transition()
                If action parameter round is i:
                   Do Solver.one_step() for i round
                Train the rl with memory set using rl_model.learn
        """
        # init state at first
        # 3BEI INTERVAL

        state = self.init_state()
        print("State NOrmalization")
        # 初始化随机探索用到的开关
        once = True
        # 初始化用来记录步数的i
        x = []
        for i in range(epoch):
            print("EPOCH {} begins".format(i))
            late_rl = time.time()
            if (i < self.random_step):
                action = [16,16,16,16]
                self.action.append(action)
                """action = np.clip([np.random.rand() * self.a_bound[0],
                                  np.random.rand() * self.a_bound[1],
                                  np.random.rand() * self.a_bound[2],
                                  np.random.rand() * self.a_bound[3],np.zeros_like(self.a_bound), self.a_bound)"""
            else:
                action_raw = self.rl_model.choose_action(state)
                action = [int(x) for x in action_raw ]
                self.action.append(action)

            late_rl = time.time() - late_rl
#             try:
            next_state, reward = self.take_action(action, late_rl)
#             except Exception as e:
#                 print(e)
#                 break

            print("STATE ACTION REWARD NEXT")
            print('Current State:', state)
            print('Action:', action)
            print('Reward:', reward)
            print('Next State:', next_state)
            # store the transition to rl_model memory
            self.rl_model.store_transition(state, action, reward, next_state)
            # store the domain adaptation tragetary
            if(self.update_RL == 1):
                self.da_memory.append([state,action,reward,next_state])

            # if train then train rl
            if self.train:
                self.rl_model.learn()
                if (i % 5 == 0):
                    self.rl_model.save_model_checkpoint()
                    print("Save RL Model")

            state = next_state

        # --optional: plot the param
        plot = True
        if plot:
            def plot_data(x_data, y_data, title=' ', x_label=' ', y_label=' ', save=False):
                # new figure
                plt.figure()
                # plot, set x, y label and title
                plt.plot(x_data, y_data)
                plt.title(title)
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                # if save then save
                if save:
                    plt.savefig('./{}.png'.format(title))
                # show the fig
                plt.show()
                plt.close(0)

            itemZip = dict(zip(self.epoch,self.accuracy_epoch))
            with open('accuracy_epoch.json', 'w', encoding='utf-8') as fs:
                json.dump(itemZip, fs)

            itemZip = dict(zip(self.epoch,self.reward))
            with open('reward_epoch.json', 'w', encoding='utf-8') as fs:
                json.dump(itemZip, fs)

            itemZip = dict(zip(self.epoch,self.fps))
            with open('flops_epoch.json', 'w', encoding='utf-8') as fs:
                json.dump(itemZip, fs)

            #self.action = [tmp.tolist() for tmp in self.action]
            itemZip = dict(zip(self.epoch,self.action))
            with open('action.json', 'w', encoding='utf-8') as fs:
                json.dump(itemZip, fs)
                """

            plot_data(self.epoch, self.accuracy_epoch, title="accuracy_epoch")
            plot_data(self.epoch, self.reward, title="reward")
            plot_data(self.epoch, self.fps, title="flops")
            plot_data(self.epoch, self.action, title="action")
            print("accuracy ")
            print(np.mean(self.accuracy_epoch))
            print("reward ")
            print(np.mean(self.reward))


            #Save DA data to file
            if(self.update_RL == 1):
                da_toFile = np.array(self.da_memory)
                np.save('da.npy',da_toFile )
            #print(da_toFile)
            print("Store Domain Adaptation Successfully!")
"""
def get_parser():
    """
    Creates an argument parser.
    """
    parser = argparse.ArgumentParser(description='AutoML for Stool Image Dataset')
    # Training parameters
    parser.add_argument('--cuda', default='1', type=str, help='CUDA visible devices')
    parser.add_argument('--epoch', default='1', type=int, help='RL epochs')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    epoch = args.epoch
    
    dis_dim = 0
    a_bound = [0, 16, 16, 16, 16]
    s_bound = [1, 1,16,16,16,16] 
    scal_dim = np.shape(a_bound[1:])[0]
    s_dim = np.shape(s_bound)[0]
    scal_var = 0.1
    train = True

    env = Environment(dis_dim=dis_dim, scal_dim=scal_dim, 
                      s_dim=s_dim, scal_var=scal_var, 
                      a_bound=a_bound, s_bound=s_bound,
                      train=train)
    env.main(epoch=epoch)
    if train:
        env.rl_model.save_model_checkpoint()

if __name__ == '__main__':
    main()