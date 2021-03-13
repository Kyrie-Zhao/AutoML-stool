import os
import time

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
    
class Net(nn.Module):
    def __init__(self, num_classes=3, num_features = 512):
        super(Net, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(num_features, num_classes, bias=True),
        )
    def forward(self, x):
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        out = self.classifier(x)
        return out
    
class StoolDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(os.path.join(root_dir, csv_file))
        self.bristol_label = list(self.annotations.bristol_type.values)
        self.condition_label = list(self.annotations.condition.values)
        self.brsitol_on_condition_label = list(self.annotations.brsitol_on_condition.values)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.loc[index, 'image_id'])
        image = Image.open(img_path).convert('RGB')
        bristol_label = torch.tensor(int(self.annotations.loc[index, 'bristol_type']))
        condition_label = torch.tensor(int(self.annotations.loc[index, 'condition']))
        brsitol_on_condition_label = torch.tensor(int(self.annotations.loc[index, 'brsitol_on_condition'])) 
        if self.transform:
            image = self.transform(image)
        return (image, bristol_label, condition_label, brsitol_on_condition_label)

def train_corse(device, net, epochs, batch_size, lr, reg, log_every_n=100, 
                log_path = 'model_checkpoints', model_name = 'coarse'):
    transform_train = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    trainset = StoolDataset(csv_file = 'train_a_b_annotation.csv', root_dir = 'dataset_v4', transform = transform_train)
    trainloader = DataLoader(trainset, sampler = BalancedBatchSampler(trainset, trainset.condition_label), 
                             batch_size = batch_size, num_workers=2)
    label_con = trainset.condition_label  
    _, sample_counts = np.unique(label_con, return_counts = True)
    num_class = len(sample_counts)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.875, weight_decay=reg, nesterov=False)
    criterion = nn.CrossEntropyLoss()
    best_bacc = 0  
    global_steps = 0
    start = time.time()
    for epoch in range(1, epochs+1):
        if epoch%2 == 0:
            lr = lr/10
        print('\nEpoch: %d' % epoch)
        train_loss = 0        
        correct = 0
        total = 0
        for batch_idx, (inputs, bristol_label, condition_label, brsitol_on_condition_label) in enumerate(trainloader):
            net.train()
            inputs, condition_label = inputs.to(device), condition_label.to(device)
            optimizer.zero_grad()
            condition_outputs = net(inputs)
            loss = CB_loss(device, condition_label, condition_outputs, sample_counts, num_class, 'softmax', 0.9999, 2.0)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, condition_predicted = condition_outputs.max(1)
            total += condition_label.size(0)
            correct += condition_predicted.eq(condition_label).sum().item()
            global_steps += 1
            if global_steps % log_every_n == 0:
                end = time.time()
                num_examples_per_second = log_every_n * batch_size / (end - start)
                print("[Step=%d]\tLoss=%.4f\tacc=%.4f\t%.1f examples/second"
                      % (global_steps, train_loss / (batch_idx + 1), (correct / total), num_examples_per_second))
                bacc, acc, _, _ , _ = test_corse(device, net, batch_size)
                if bacc > best_bacc:
                    best_bacc = bacc
                    print("Saving...")
                    torch.save(net.state_dict(), os.path.join(log_path, model_name))
                start = time.time()
        
def test_corse(device, net, batch_size):
    transform_test = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    testset = StoolDataset(csv_file = 'test_a_annotation.csv', 
                           root_dir = 'dataset_v4', transform = transform_test)
    testloader = DataLoader(testset, batch_size = batch_size, num_workers = 2)
    net.eval()
    test_loss = 0     
    y_pred = []
    y_true = []
    list_outputs = []
    with torch.no_grad():
        for batch_idx,(inputs, bristol_label, condition_label, brsitol_on_condition_label) in enumerate(testloader):
            inputs, condition_label = inputs.to(device), condition_label.to(device)  
            condition_outputs = net(inputs)
            list_outputs.append(condition_outputs.cpu().detach().numpy())
            _, condition_predicted = condition_outputs.max(1)
            y_pred = y_pred + condition_predicted.tolist()
            y_true = y_true + condition_label.tolist()
    y_output = np.concatenate(list_outputs, axis=0)
    num_val_steps = len(testloader)
    bacc = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    print("Test acc=%.4f, Test balanced accuracy=%.4f" % (acc, bacc))
    return bacc, acc, y_true, y_output, y_pred

def train_fine(device, net_coarse, net_fine_0, net_fine_1, net_fine_2, positions, condition, 
               epochs, batch_size, lr, reg, log_every_n=100, log_path = 'model_checkpoints', model_name = 'fine'):
    transform_train = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    trainset = StoolDataset(csv_file = 'train_a_b_annotation.csv', root_dir = 'dataset_v4', transform = transform_train)
    trainloader = DataLoader(trainset, sampler = BalancedBatchSampler(trainset, trainset.bristol_label), 
                             batch_size = batch_size, num_workers=2)
    label_con = [element for idx, element in enumerate(trainset.brsitol_on_condition_label) if
                 trainset.condition_label[idx]==condition]
    _, sample_counts = np.unique(label_con, return_counts = True)
    num_class = len(sample_counts) 
    net = torchvision.models.mobilenet_v2(pretrained=True)
    
    optimizer = optim.SGD(net_fine.parameters(), lr=lr, momentum=0.875, weight_decay=reg, nesterov=False)
    best_bacc = 0  
    global_steps = 0
    start = time.time()
    for epoch in range(1, epochs+1):
        print('\nEpoch: %d' % epoch)
        net_coarse.eval()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, _, con_label, brsitol_con_label) in enumerate(trainloader):
            net_fine.train()
            inputs, con_label = inputs.to(device), con_label.to(device)
            brsitol_con_label = brsitol_con_label.to(device)
            inputs, labels = inputs[con_label==condition], brsitol_con_label[con_label==condition]
            optimizer.zero_grad()
            modules = list(net_coarse.features.children())
            modules = modules[:position+1]
            
            extracter = nn.Sequential(*modules)
            features = extracter(inputs) 
            outputs = net_fine(features)
            loss = CB_loss(device, labels, outputs, sample_counts, num_class, 'softmax', 0.9999, 2.0)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            global_steps += 1
            if global_steps % log_every_n == 0:
                end = time.time()
                num_examples_per_second = log_every_n * batch_size / (end - start)
                print("[Step=%d]\tLoss=%.4f\tacc=%.4f\t%.1f examples/second"
                      % (global_steps, train_loss / (batch_idx + 1), (correct / total), num_examples_per_second))
                start = time.time()
                bacc, acc = test_fine(device, net_coarse, net_fine, position, batch_size, condition)
                if bacc > best_bacc:
                    best_bacc = bacc
                    print("Saving...")
                    torch.save(net_fine.state_dict(), os.path.join(log_path, model_name))
            
def test_fine(device, net_coarse, net_fine, position, batch_size, condition):
    transform_test = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    testset = StoolDataset(csv_file = 'test_a_annotation.csv', 
                           root_dir = 'dataset_v4', transform = transform_test)
    testloader = DataLoader(testset, batch_size = batch_size, num_workers = 2)
    net_coarse.eval()
    net_fine.eval()
    test_loss = 0     
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch_idx, (inputs, _, con_label, brsitol_con_label) in enumerate(testloader):
            inputs, con_label = inputs.to(device), con_label.to(device); 
            brsitol_con_label = brsitol_con_label.to(device)
            inputs, labels = inputs[con_label==condition], brsitol_con_label[con_label==condition]
            if len(labels) == 0:
                continue
            modules = list(net_coarse.features.children())
            modules = modules[:position+1]
            extracter = nn.Sequential(*modules)
            features = extracter(inputs) 
            outputs = net_fine(features)
            _, predicted = outputs.max(1)
            y_pred = y_pred + predicted.tolist()
            y_true = y_true + labels.tolist()
    bacc = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    print("Test acc=%.4f, Test balanced accuracy=%.4f" % (acc, bacc))
    return bacc, acc

def test_full(device, net_coarse, net_fine_0, net_fine_1, net_fine_2, positions, batch_size):
    transform_test = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    testset = StoolDataset(csv_file = 'test_a_annotation.csv', 
                           root_dir = 'dataset_v4', transform = transform_test)
    testloader = DataLoader(testset, batch_size = batch_size, num_workers = 2)
    net_coarse.eval()
    net_fine_0.eval()
    net_fine_1.eval()
    net_fine_2.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch_idx, (inputs, brsitol_label, _, _) in enumerate(testloader):
            inputs, brsitol_label = inputs.to(device), brsitol_label.to(device); 
            outputs_coarse = net_coarse(inputs)
            _, predicted_coarse = outputs_coarse.max(1)
            predicted_fine = torch.empty(brsitol_label.size(), dtype = torch.long, device = device)
            modules = list(net_coarse.features.children())
            modules_0 = modules[:positions[1]+1]
            features_0 = nn.Sequential(*modules_0)(inputs[predicted_coarse == 0]) 
            modules_1 = modules[:positions[2]+1]
            features_1 = nn.Sequential(*modules_1)(inputs[predicted_coarse == 1]) 
            modules_2 = modules[:positions[3]+1]
            features_2 = nn.Sequential(*modules_2)(inputs[predicted_coarse == 2]) 

            if len(features_0) != 0:
                outputs_fine_0 = net_fine_0(features_0)
                _, predicted_fine_0 = outputs_fine_0.max(1)
                predicted_fine[predicted_coarse==0] = predicted_fine_0
            
            if len(features_1) != 0:   
                outputs_fine_1 = net_fine_1(features_1) 
                _, predicted_fine_1 = outputs_fine_1.max(1)
                predicted_fine[predicted_coarse==1] = predicted_fine_1 + 2
            
            if len(features_2) != 0:   
                outputs_fine_2 = net_fine_2(features_2)
                _, predicted_fine_2 = outputs_fine_2.max(1)
                predicted_fine[predicted_coarse==2] = predicted_fine_2 + 5
                
            y_pred = y_pred + predicted_fine.tolist()
            y_true = y_true + brsitol_label.tolist()
    bacc = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    print("Test acc=%.4f, Test balanced accuracy=%.4f" % (acc, bacc))
    return bacc, acc, y_pred, y_true