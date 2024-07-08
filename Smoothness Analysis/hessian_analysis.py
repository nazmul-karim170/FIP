##
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
##

from __future__ import print_function

import json
import os
import sys

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from utils_hessian import *
from density_plot import get_esd_plot
from pyhessian import hessian
import models
import glob 

from dataloader_cifar import * 
import poison_cifar as poison

## Settings
parser = argparse.ArgumentParser(description='PyTorch Example')

parser.add_argument(
    '--mini-batch-size',
    type=int,
    default=200,
    help='input batch size for mini-hessian batch (default: 200)')

parser.add_argument('--batch-size',
                    type=int,
                    default=200,
                    help='input batch size for hessian (default: 200)')

parser.add_argument('--seed',
                    type=int,
                    default=1,
                    help='random seed (default: 1)')

parser.add_argument('--batch-norm',
                    action ='store_false',
                    default = True,
                    help='do we need batch norm or not')

parser.add_argument('--residual',
                    action='store_false',
                    default= True,
                    help='do we need residual connect or not')

parser.add_argument('--cuda',
                    action='store_false',
                    default = True,
                    help='do we use gpu or not')

parser.add_argument('--train_mode',
                    type=str,
                    default = '',
                    help='Backdoor or Purified')

parser.add_argument('--depth',
                    type=int,
                    default =18,
                    help='ResNet Architecture Depth')

parser.add_argument('--dataset',
                    default='CIFAR10',
                    help='Dataset We have Used')

parser.add_argument('--data_type',
                    default='clean',
                    help='Clean or Trojan Data')

parser.add_argument('--resume',
                    type=str,
                    default='',
                    help='get the checkpoint')

parser.add_argument('--mode',
                    type=str,
                    default='clean',
                    help='For Plotting')

                                ## For Trojan Data
parser.add_argument('--poison-type', type=str, default='badnets', choices=['badnets', 'Feature', 'FC',  'SIG', 'Dynamic', 'TrojanNet', 'blend', 'CLB', 'benign'],
                    help='type of backdoor attacks used during training')
parser.add_argument('--poison-rate', type=float, default=1, help='proportion of poison examples in the training set')
parser.add_argument('--poison-target', type=int, default=0, help='target class of backdoor attack')
parser.add_argument('--trigger-alpha', type=float, default=0.2, help='the transparency of the trigger pattern.')
parser.add_argument('--val_frac', type=float, default=0, help='ratio of validation samples')
parser.add_argument('--target_label', type=int, default=0, help='class of target label')
parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
parser.add_argument('--trig_w', type=int, default=3, help='width of trigger pattern')
parser.add_argument('--trig_h', type=int, default=3, help='height of trigger pattern')
parser.add_argument('--data-dir',   type=str, default='./dataset', help='dir to the dataset')
parser.add_argument('--output-dir', type=str, default='logs/models/')
parser.add_argument('--save_file', type=str, default='eigenvalues.npz')
parser.add_argument('--ACC', type=float, default=95.21, help='width of trigger pattern')
parser.add_argument('--ASR', type=float, default=0, help='height of trigger pattern')
parser.add_argument('--gpuid', type=int, default=1, help='the transparency of the trigger pattern.')

## Basic Model Parameters.
## Basic Model Parameters.
parser.add_argument('--arch', type=str, default='resnet18',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'MobileNetV2', 'vgg19', 'dense', 'google', 'inception'])
parser.add_argument('--widen-factor', type=int, default=1, help='Widen_Factor for WideResNet')

args = parser.parse_args()

## set random seed to reproduce the work
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

for arg in vars(args):
    print(arg, getattr(args, arg))

# torch.cuda.set_device(args.gpuid)

## Get Dataset
# if args.data_type == 'clean':
#     train_loader, test_loader = getData(name='cifar10_without_dataaugmentation',
#                                     train_bs=args.mini_batch_size,
#                                     test_bs=1)
# else:

def test(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.cuda(), labels.cuda()
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = torch.max(output,1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc

os.makedirs(args.output_dir, exist_ok=True)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=1),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
])

transform_none = transforms.ToTensor()
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
])

## Step 1: Create Poisoned / Clean dataset
orig_train             = CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
clean_train, clean_val = poison.split_dataset(dataset=orig_train, val_frac=args.val_frac,
                                              perm=np.loadtxt('./data/cifar_shuffle.txt', dtype=int))

clean_test    = CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
test_loader   = DataLoader(clean_test, batch_size=args.batch_size, num_workers=4)

triggers = {'badnets': 'checkerboard_1corner',
            'CLB': 'fourCornerTrigger',
            'blend': 'gaussian_noise',
            'SIG': 'signalTrigger',
            'TrojanNet': 'trojanTrigger',
            'FC': 'gridTrigger',
            'benign': None}

if args.poison_type == 'badnets':
    args.trigger_alpha = 0.6
elif args.poison_type == 'blend':
    args.trigger_alpha = 0.2

if args.poison_type in ['badnets', 'blend']:
    trigger_type      = triggers[args.poison_type]
    args.trigger_type = trigger_type
    
    if args.train_mode == 'purified':
        poison_train, trigger_info = \
            poison.add_trigger_cifar_true_label(data_set=clean_train, trigger_type=trigger_type, poison_rate=args.poison_rate,
                                     poison_target=args.poison_target, trigger_alpha=args.trigger_alpha)
        print("For Purified Model")
    
    else:
        poison_train, trigger_info = \
            poison.add_trigger_cifar(data_set=clean_train, trigger_type=trigger_type, poison_rate=args.poison_rate,
                                     poison_target=args.poison_target, trigger_alpha=args.trigger_alpha)

    poison_test = poison.add_predefined_trigger_cifar(data_set=clean_test, trigger_info=trigger_info)
    poison_train_loader = DataLoader(poison_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    poison_test_loader  = DataLoader(poison_test,  batch_size=args.batch_size, num_workers=4)
    clean_test_loader   = DataLoader(clean_test,   batch_size=args.batch_size, num_workers=4)
    ASR = 100

elif args.poison_type in ['Dynamic']:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])        
    
    ## Load the fixed poisoned data, e.g. Dynamic. (This is bit complicated, needs some pre-defined tasks)
    poisoned_data = Dataset_npy(np.load(args.poisoned_data_train, allow_pickle=True), transform = transform_train)
    poison_train_loader = DataLoader(dataset=poisoned_data,
                                    batch_size=args.batch_size,
                                    shuffle=True)

    poisoned_data = Dataset_npy(np.load(args.poisoned_data_test, allow_pickle=True), transform = transform_test)
    poison_test_loader = DataLoader(dataset=poisoned_data,
                                    batch_size=args.batch_size,
                                    shuffle=True)
    clean_test_loader  = DataLoader(clean_test, batch_size=args.batch_size, num_workers=4)
    trigger_info = None
    ASR = 100

## For clean Label attacks, Provided Implementation Gives Good ASR. Failure to obtain that may require Adverarial Perturbations 
elif args.poison_type in ['SIG', 'TrojanNet', 'CLB']:
    trigger_type      = triggers[args.poison_type]
    args.trigger_type = trigger_type      
    args.inject_portion = args.poison_rate  

    ## SIG and CLB are Clean-Label Attacks 
    if args.poison_type in ['SIG', 'CLB']:
        args.target_type = 'cleanLabel'

    poisoned_data, poison_train_loader = get_backdoor_loader(args)
    _, poison_test_loader = get_test_loader(args)
    clean_test_loader     = DataLoader(clean_test, batch_size=args.batch_size, num_workers=4)
    trigger_info          = None
    ASR = 100

else:
    poison_train_loader = DataLoader(clean_train, batch_size=args.batch_size, num_workers=4)

train_loader = DataLoader(clean_train, batch_size=args.batch_size, num_workers=4)

##############
## Get the Hessian Data
##############
assert (args.batch_size % args.mini_batch_size == 0)
assert (50000 % args.batch_size == 0)
batch_num = args.batch_size // args.mini_batch_size

if batch_num == 1:
    for inputs, labels in train_loader:
        hessian_dataloader = (inputs, labels)
        break
else:
    hessian_dataloader = []
    for i, (inputs, labels) in enumerate(train_loader):
        hessian_dataloader.append((inputs, labels))
        if i == batch_num - 1:
            break

## Get Model
# model = getattr(models, args.arch)(num_classes=10)
criterion = nn.CrossEntropyLoss()                    ## Label Loss

top_eigens = []
Tr_H  = []

save_file = "eigenvalues_trace_new_2_" + str(args.poison_type)+ ".npz"
print("Loading File", save_file)
    
###################
## Get Model Checkpoint, get saving folder
###################
model = getattr(models, args.arch)(num_classes=10)
if args.resume == '':
    raise Exception("Please choose the trained model")
model.load_state_dict(torch.load(args.resume, map_location='cpu'))

if args.cuda:
    model = model.cuda()
model = torch.nn.DataParallel(model)

######################################################
## Validate the model 
######################################################

cl_test_loss, cl_test_acc = test(model=model, criterion=criterion, data_loader=test_loader)
print("Validation Accuracy of the Given Model:", cl_test_acc)

# cl_test_loss, cl_test_acc = test(model=model, criterion=criterion, data_loader=train_loader)
# print("Train Accuracy of the Given Model:", cl_test_acc)

if args.poison_type != 'benign':
  po_test_loss, po_test_acc = test(model=model, criterion=criterion, data_loader=poison_test_loader)
  print("Attack Success Rate of the Given Model:", po_test_acc)


######################################################
# Begin the computation
######################################################

## Turn model to eval mode
if os.path.exists(os.path.join(args.output_dir,save_file)):
    density_eigen   = np.load(os.path.join(args.output_dir,save_file))['density_eigen']
    density_weight  = np.load(os.path.join(args.output_dir,save_file))['weight']
    top_eigenvalues = np.load(os.path.join(args.output_dir,save_file))['eignevalues']
    trace =  np.load(os.path.join(args.output_dir, save_file))['trace']

else:
    model.eval()
    if batch_num == 1:
        hessian_comp = hessian(model,
                               criterion,
                               data=hessian_dataloader,
                               cuda=args.cuda)
    else:
        hessian_comp = hessian(model,
                               criterion,
                               dataloader=hessian_dataloader,
                               cuda=args.cuda)

    print('********** Finished Data Loading and Begin Hessian Computation **********')
    top_eigenvalues, _ = hessian_comp.eigenvalues()
    trace              = hessian_comp.trace()
    # print("Print Trace:", trace)
    density_eigen, density_weight = hessian_comp.density()

## Save the statistics in a numpy file
np.savez(os.path.join(args.output_dir,save_file), density_eigen = density_eigen, weight = density_weight, eignevalues = np.mean(np.array(top_eigenvalues)), trace=np.mean(trace))

print('\n***Top Eigenvalue: ', top_eigenvalues)
print('\n***Trace of Hessian: ', np.mean(trace))
get_esd_plot(density_eigen, density_weight, np.mean(np.array(top_eigenvalues)), np.mean(trace), args.ACC, args.ASR , args.mode, args.output_dir, 'Eigen_')


