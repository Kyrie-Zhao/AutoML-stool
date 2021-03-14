import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader 

import torchvision
import torchvision.transforms as transforms

from MODEL.torchsampler import ImbalancedDatasetSampler, BalancedBatchSampler
from MODEL.class_balanced_loss import CB_loss
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, accuracy_score

import os
from MODEL.train_util import train_coarse, test_coarse, train_fine, test_fine, test_full, StoolDataset
from MODEL.train_util import Net
from MODEL.evaluate_util import plot_confusion_matrix
import numpy as np
import pandas as pd
from skimage import io
from PIL import Image

from sklearn.metrics import balanced_accuracy_score, accuracy_score, cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

class Trainer(object):

    def __init__(self, device):
        self.data_root = 'dataset_v4'
        self.log_path = 'model_checkpoints'
        self.device = device
        self.epochs_coarse = 2
        self.epochs_fine = 1
        self.batch_size = 42
        self.lr_coarse = 0.1
        self.lr_fine = 0.1
        self.reg = 1e-5
        self.log_every_n = 50
        self.num_features = [32, 16, 24, 24, 32, 32, 32, 64, 64, 64, 64,
                              96, 96, 96, 160, 160, 160, 320, 1280]
        self.Flops = [10.84, 20.87, 50.08, 75.82, 91.29, 102.28, 113.27, 120.84, 131.16, 
                      141.48, 151.8, 164.53, 187.23, 209.93, 225.54, 241.02, 256.5, 279.5, 299.57, 300.85]
        self.num_train = [51, 2278, 394]
                
    def train(self, positions):
        print('Current positions:', positions)
        start_time = time.time()
        net_base = torchvision.models.mobilenet_v2(pretrained=True)
        net_base.classifier = torch.nn.Linear(self.num_features[-1], out_features=3)
        net_base.load_state_dict(torch.load(os.path.join(self.log_path, 'coarse_pre_train')))
        net_base = net_base.to(self.device)
        
        net_coarse = Net(num_classes=3, num_features = self.num_features[positions[0]]).to(self.device)
        net_fine_0 = Net(num_classes=2, num_features = self.num_features[positions[1]]).to(self.device)
        net_fine_1 = Net(num_classes=3, num_features = self.num_features[positions[2]]).to(self.device)
        net_fine_2 = Net(num_classes=2, num_features = self.num_features[positions[3]]).to(self.device)
                
        bacc_coarse = train_coarse(self.device, net_base, net_coarse, positions[0], 
                                   self.epochs_coarse, self.batch_size, self.lr_coarse, self.reg, 
                                   self.log_every_n, self.log_path, model_name = 'test')
        if bacc_coarse < 0.4:
            bacc = 0.14
        else:
            train_fine(self.device, net_base, net_fine_0, positions[1], 0, self.epochs_fine, self.batch_size, 
                       self.lr_fine, self.reg, self.log_every_n, self.log_path, model_name = 'test0')
            train_fine(self.device, net_base, net_fine_1, positions[2], 1, self.epochs_fine, self.batch_size, 
                       self.lr_fine, self.reg, self.log_every_n, self.log_path, model_name = 'test1')
            train_fine(self.device, net_base, net_fine_2, positions[3], 2, self.epochs_fine, self.batch_size, 
                       self.lr_fine, self.reg, self.log_every_n, self.log_path, model_name = 'test2')

            net_coarse.load_state_dict(torch.load(os.path.join(self.log_path, 'test')))
            net_fine_0.load_state_dict(torch.load(os.path.join(self.log_path, 'test0')))
            net_fine_1.load_state_dict(torch.load(os.path.join(self.log_path, 'test1')))
            net_fine_2.load_state_dict(torch.load(os.path.join(self.log_path, 'test2')))

            bacc, acc, y_pred_a, y_true_a = test_full(self.device, net_base, net_coarse, net_fine_0, 
                                                      net_fine_1, net_fine_2, positions, self.batch_size)
            
        flops = self.Flops[positions[0]]*(51+2278+394)
        for i in range(1, len(positions)):
            if positions[i] > positions[0]:
                flops +self.Flops[positions[i]]*self.num_train[i-1]
        flops = flops / (51+2278+394)
        print("### One RL epoch costs %s seconds ###" %(time.time() - start_time))
        return [bacc, flops]

def get_parser():
    """
    Creates an argument parser.
    """
    parser = argparse.ArgumentParser(description='Trainer for Stool Image Dataset')
    parser.add_argument('--cuda', default='1', type=str, help='CUDA visible devices')
    parser.add_argument('--epoch', default='1', type=int, help='RL epochs')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    trainer = Trainer(device=torch.device('cuda:'+args.cuda))
    trainer.train([12, 15, 15, 15])
    
if __name__ == '__main__':
    main()
