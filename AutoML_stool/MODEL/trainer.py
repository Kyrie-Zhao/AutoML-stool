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
# from resnet20 import ResNetCIFAR
from MODEL.train_util import train_corse, test_corse, train_fine, test_fine, test_full, StoolDataset
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
        self.epochs_coarse = 6
        self.epochs_fine = 1
        self.batch_size = 14
        self.lr_coarse = 0.01
        self.lr_fine = 0.001
        self.reg = 1e-5
        self.log_every_n = 100
        self.dict_features = {0:32, 1:16, 2:24, 3:24, 4:32, 5:32, 6:32, 7:64, 8:64, 9:64, 10:64,
                              11:96, 12:96, 13:96, 14:160, 15:160, 16:160, 17:320, 18:1280}
        self.Flops = [10.84, 20.87, 50.08, 75.82, 91.29, 102.28, 113.27, 120.84, 131.16, 
                      141.48, 151.8, 164.53, 187.23, 209.93, 225.54, 241.02, 256.5, 279.5, 299.57, 300.85]
                
    def train(self, positions):
        print('positions:', positions)
        start_time = time.time()
        net = torchvision.models.mobilenet_v2(pretrained=True)
        print(net.features[0])
        modules = list(net.features.children())
        modules = modules[:positions[0]+1]
        extracter = nn.Sequential(*modules)
        net.features = extracter
        net.classifier = torch.nn.Linear(self.dict_features[positions[0]], out_features=3)
        net = net.to(self.device)
        train_corse(self.device, net, self.epochs_coarse, self.batch_size, self.lr_coarse, self.reg, 
                    self.log_every_n, self.log_path, model_name = 'coarse_14_11_5_10')
        
        net_constipation = Net(num_classes=2, num_features = self.dict_features[positions[1]]).to(self.device)
        net_normal = Net(num_classes=3, num_features = self.dict_features[positions[2]]).to(self.device)
        net_loose = Net(num_classes=2, num_features = self.dict_features[positions[3]]).to(self.device)
                    
        train_fine(self.device, net, net_constipation, positions[1], 0, self.epochs_fine, self.batch_size, 
                   self.lr_fine, self.reg, self.log_every_n, self.log_path, model_name = 'fine0_14_11_5_10')
        train_fine(self.device, net, net_normal, positions[2], 1, self.epochs_fine, self.batch_size, self.lr_fine, self.reg, 
                   self.log_every_n, self.log_path, model_name = 'fine1_14_11_5_10')
        train_fine(self.device, net, net_loose, positions[3], 2, self.epochs_fine, self.batch_size, self.lr_fine, self.reg, 
                   self.log_every_n, self.log_path, model_name = 'fine2_14_11_5_10')
            
        net.load_state_dict(torch.load(os.path.join(self.log_path, 'coarse_14_11_5_10')))
        net_constipation.load_state_dict(torch.load(os.path.join(self.log_path, 'fine0_14_11_5_10')))
        net_normal.load_state_dict(torch.load(os.path.join(self.log_path, 'fine1_14_11_5_10')))
        net_loose.load_state_dict(torch.load(os.path.join(self.log_path, 'fine2_14_11_5_10')))
        
        bacc, acc, y_pred_a, y_true_a = test_full(self.device, net, net_constipation, 
                                                  net_normal, net_loose, positions, self.batch_size)
        flops = self.Flops[positions[0]]
        print("### One RL epoch costs %s seconds ###" %(time.time() - start_time))
        return [bacc, flops*(10**6)]
    
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
    trainer.train([14, 11, 5, 10])
    
if __name__ == '__main__':
    main()
