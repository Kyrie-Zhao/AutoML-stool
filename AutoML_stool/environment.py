import os
import time
import psutil
import pickle
import numpy as np
import tensorflow as tf
import json
from RL_AGENT.ddpg import Ddpg
import matplotlib.pyplot as plt
from MODEL.Client.solver import Solver

class Environment(object):
    """
     The Main Environment to connect solver with rl
    """
    def __init__(self, dis_dim, scal_dim, s_dim, scal_var, a_bound, s_bound, train=True):
        """
        init solver and rl_model
        :param train: Train rl or not
        :param s_bound: State's boundary if s range is [[0-1],[0-9],[0-100]] then s_bound [0, 9, 100]
        :param solver: Solver class, which have .one_step
        :param rl_model: rl_model class, which have .choose_action, .store_transition and .learn
        """
        # da or host 0 = host 1 = da
        self.update_RL = 1
        # init param for rl
        self.dis_dim = dis_dim
        self.scal_dim = scal_dim
        self.s_dim = s_dim
        self.scal_var = scal_var
        self.a_bound = a_bound
        self.s_bound = s_bound
        self.da_memory = []
        # init train and bound
        self.train = train
        # init power and late user constrain

        # init solver and rl
        self.solver = Solver()
        print("3")
        self.rl_model = Ddpg(self.dis_dim, self.scal_dim, self.s_dim, self.scal_var, self.a_bound)
        print("Solver Loading")

        if (self.flag_DA == 1):
            da_extractedData = np.load('da.npy', allow_pickle=True).tolist()
            for i in range(0,len(da_extractedData)):
                self.rl_model.store_transition(da_extractedData[i][0],da_extractedData[i][1],da_extractedData[i][2],da_extractedData[i][3])
            print("Restore Domain Adaptation Successfully!")
        # 随机探索的步数
        self.random_step = 1
        # 全局的step
        self.global_step = 0
        self.global_epoch = 0

        self.init_param_to_store()


    def init_param_to_store(self):
        self.step = []
        self.accuracy = []
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
        print("init_state")
        mean = np.multiply(self.s_bound, 0.5)
        std = np.multiply(self.s_bound, 0.1)
        # generate normal
        state = np.random.normal(mean, std)
        print(state)

        return state


    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    def calculate_p(self, acc, fps):
        """
        """
        if acc < 0.8:
            reward = -1
        else:
            reward = (1-sigmoid(fps))
        return reward



    def calculate_reward(self, state):
        """

        """
        # state_dim = [acc, latency, power]
        acc, fps = state[0], state[1]
        # calculate reward
        reward = self.calculate_p(acc, fps)

        return reward

    def take_action(self, action, late_rl):
        """
        Used function:
            solver.one_step => a function mapping action to new state
            calculate_reward => a function mapping state to reward

        """
        # 全局step加1，用于标志数据收集
        print("action")
        print(action)
        self.global_epoch += 1

        # init next_state, to_average_state
        next_state_to_average = []
        out_point_avg = []

        self.global_step += 1

        #~~~~~~~~~~~``[acc,f\ps]
        next_state = self.solver.one_step(action.tolist())
        print(next_state)
        # 收集相关数据(step)
        self.step.append(self.global_step)

        self.accuracy.append(next_state[0])
        self.fps.append(next_state[1])


        # calculate reward
        reward = self.calculate_reward(next_state)

        # 收集相关数据:
        self.epoch.append(self.global_epoch)
        self.accuracy_epoch.append(next_state[0])
        self.fps_epoch.append(next_state[1])
        self.reward.append(reward)

        # 返回rl的观测数据
        return np.array([next_state[0], next_state[1]]), reward

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
            # rl记录时间
            late_rl = time.time()
            # 随机探索
            if (i < self.random_step):
                if once:
                    self.a_bound = [self.a_bound[0],self.a_bound[1], self.a_bound[2], self.a_bound[3]]
                    once = False
                action = [5,7,9,7]
                """action = np.clip([np.random.rand() * self.a_bound[0],
                                  np.random.rand() * self.a_bound[1],
                                  np.random.rand() * self.a_bound[2],
                                  np.random.rand() * self.a_bound[3],np.zeros_like(self.a_bound), self.a_bound)"""
            # 不随机探索
            else:
                action = self.rl_model.choose_action(state)

            late_rl = time.time() - late_rl
            print("Action Choose")
            print(action)
            # take action according to the action, get reward and next_state
            try:
                next_state, reward = self.take_action(action, late_rl)
                print("next state values")
                print(next_state)
            except Exception as e:
                print(e)
                break

            print("Big interval state")
            print(state)
            # store the transition to rl_model memory
            self.rl_model.store_transition(state, action, reward, next_state)
            # store the domain adaptation tragetary
            if(self.update_RL == 1):
                self.da_memory.append([state,action,reward,next_state])

            # if train then train rl
            if self.train:
                self.rl_model.learn()
                if (i % 500 == 0):
                    env.rl_model.save_model_checkpoint()
                    print("Store RL")


            # --optimal: print evaluate
            print_eval = True
            if print_eval:
                # set up evaluation to print
                to_print = "Epoch: {} |" \
                           " State: {} |" \
                           " Action: {} |" \
                           " Reward: {} |" \
                           " Next state: {} |".format(i, state, action, reward, next_state)
                # print no new line or just print
                print(to_print)

            state = next_state

        # --optional: plot the param
        plot = True
        if plot:
            def plot_data(x_data, y_data, title=' ', x_label=' ', y_label=' ', save=False):
                """plot data"""
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
            with open('results/accuracy_epoch.json', 'w', encoding='utf-8') as fs:
                json.dump(itemZip, fs)

            itemZip = dict(zip(self.epoch,self.reward))
            with open('results/reward_epoch.json', 'w', encoding='utf-8') as fs:
                json.dump(itemZip, fs)

            plot_data(self.epoch, self.accuracy_epoch, title="accuracy_epoch")
            plot_data(self.epoch, self.reward, title="reward")
            plot_data(self.step, self.accuracy, title="accuracy")

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



if __name__ == '__main__':
    # when use:
    """
    action[]
    """
    dis_dim = 0
    #[Coarse position, fine position]
    a_bound = [16, 16, 16, 16]
    #bandwidth [FLOPS, ACC]
    s_bound = [1, 1]
    scal_dim = np.shape(a_bound[1:])[0]
    s_dim = np.shape(s_bound)[0]
    scal_var = 0.1
    #pow_u = 6.4
    #late_u = 1
    train = True

    # init env
    env = Environment(dis_dim=dis_dim,
                      scal_dim=scal_dim,
                      s_dim=s_dim,
                      scal_var=scal_var,
                      a_bound=a_bound,
                      s_bound=s_bound,
                      train=train)

    # for epoch
    epoch = 10

    # run main
    env.main(epoch=epoch)
    #env.store_all()
    if train:
        env.rl_model.save_model_checkpoint()
