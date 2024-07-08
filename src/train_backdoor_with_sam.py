import os
import time
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from data_loader import * 
from config import get_arguments
import networks
import poison_cifar as poison
from PIL import Image
from data_loader import *
import random
import torch.nn as nn

from sam_model.wide_res_net import WideResNet
from sam_model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from sam import SAM

parser = argparse.ArgumentParser(description='Train poisoned networks')
from Regularizer import CDA_Regularizer as regularizer   ## Regularizer 


## Basic Model Parameters.
parser.add_argument('--arch', type=str, default='resnet18',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'MobileNetV2', 'vgg19_bn'])
parser.add_argument('--widen-factor', type=int, default=1, help='Widen_Factor for WideResNet')
parser.add_argument('--batch-size', type=int, default=256, help='the batch size for dataloader')
parser.add_argument('--epoch',      type = int, default = 250, help='the numbe of epoch for training')
parser.add_argument('--schedule',   type=int, nargs='+', default=[30, 100, 150], help='Decrease learning rate at these epochs.')
parser.add_argument('--save-every', type=int, default=2, help='save checkpoints every few epochs')
parser.add_argument('--data-dir',   type=str, default='../data', help='dir to the dataset')
parser.add_argument('--output-dir', type=str, default='logs/models/')
parser.add_argument('--checkpoint', type=str, help='The checkpoint to be pruned')

## Backdoor Parameters
parser.add_argument('--clb-dir', type=str, default='', help='dir to training data under clean label attack')
parser.add_argument('--poison-type', type=str, default='badnets', choices=['badnets', 'FC',  'SIG', 'Dynamic', 'TrojanNet', 'blend', 'CLB', 'benign'],
                    help='type of backdoor attacks used during training')
parser.add_argument('--poison-rate', type=float, default=0.10, help='proportion of poison examples in the training set')
parser.add_argument('--poison-target', type=int, default=0, help='target class of backdoor attack')
parser.add_argument('--trigger-alpha', type=float, default=0.2, help='the transparency of the trigger pattern.')
parser.add_argument('--gpuid', type=int, default=1, help='the transparency of the trigger pattern.')

parser.add_argument('--log_root', type=str, default='./logs', help='logs are saved here')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='name of image dataset')
parser.add_argument('--load_fixed_data', type=int, default=0, help='load the local poisoned dataest')

## Training Hyper-Parameters
parser.add_argument('--print_freq', type=int, default=200, help='frequency of showing training results on console')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--num_class', type=int, default=10, help='number of classes')
parser.add_argument('--isolation_ratio', type=float, default=0.01, help='ratio of isolation data')

## Others
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--val_frac', type=float, default=0.10, help='ratio of validation samples')
parser.add_argument('--target_label', type=int, default=0, help='class of target label')
parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
parser.add_argument('--trig_w', type=int, default=3, help='width of trigger pattern')
parser.add_argument('--trig_h', type=int, default=3, help='height of trigger pattern')

parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
parser.add_argument("--rho", default=2.0, type=int, help="Rho parameter for SAM.")
parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")


args = parser.parse_args()
args_dict = vars(args)

random.seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.cuda.set_device(args.gpuid)


def main():    
    ## Step 0: Data Transformation 
    args.output_dir = os.path.join(args.output_dir, "output_" + str(args.poison_rate) + "_" + str(args.rho) )
    os.makedirs(args.output_dir, exist_ok=True)
    logger = logging.getLogger(__name__)

    log_file = "output_" + str(args.poison_rate) + "_" + str(args.rho) + '.log'
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir,log_file)),
            logging.StreamHandler()
        ])
    logger.info(args)


    MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
    STD_CIFAR10  = (0.2023, 0.1994, 0.2010)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])

    transform_none = transforms.ToTensor()
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])

    ## Step 1: Create poisoned / Clean dataset
    orig_train = CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    clean_train, clean_val = poison.split_dataset(dataset=orig_train, val_frac=args.val_frac,
                                                  perm=np.loadtxt('./data/cifar_shuffle.txt', dtype=int))
    clean_test = CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)

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
        poison_train, trigger_info = \
            poison.add_trigger_cifar(data_set=clean_train, trigger_type=trigger_type, poison_rate=args.poison_rate,
                                     poison_target=args.poison_target, trigger_alpha=args.trigger_alpha)
        poison_test = poison.add_predefined_trigger_cifar(data_set=clean_test, trigger_info=trigger_info)
        poison_train_loader = DataLoader(poison_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
        poison_test_loader  = DataLoader(poison_test, batch_size=args.batch_size, num_workers=4)
        clean_test_loader   = DataLoader(clean_test, batch_size=args.batch_size, num_workers=4)

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
        clean_test_loader   = DataLoader(clean_test, batch_size=args.batch_size, num_workers=4)
        trigger_info = None

    ## For clean Label attacks, provided implementation gives good ASR. Failure to obtain that may require adverarial perturbations 
    elif args.poison_type in ['SIG', 'TrojanNet', 'CLB']:
        trigger_type      = triggers[args.poison_type]
        args.trigger_type = trigger_type        

        ## SIG and CLB are Clean-label Attacks 
        if args.poison_type in ['SIG', 'CLB']:
            args.target_type = 'cleanLabel'

        poisoned_data, poison_train_loader = get_backdoor_loader(args)
        _, poison_test_loader = get_test_loader(args)
        clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=4)

        trigger_info = None

    elif args.poison_type == 'benign':
        poison_train = clean_train
        poison_test = clean_test
        poison_train_loader = DataLoader(poison_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
        poison_test_loader  = DataLoader(poison_test, batch_size=args.batch_size, num_workers=4)
        clean_test_loader   = DataLoader(clean_test, batch_size=args.batch_size, num_workers=4)
        trigger_info = None
    else:
        raise ValueError('Please use valid backdoor attacks: [badnets | blend | CLB]')

    ## Step 2: prepare model, criterion, optimizer, and learning rate scheduler.
    net = getattr(networks, args.arch)(num_classes=10).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(state_dict)

    base_optimizer = torch.optim.SGD
    optimizer = SAM(net.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)

    ## Step 3: Train Backdoored Models
    logger.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_init.th'))
    if trigger_info is not None:
        torch.save(trigger_info, os.path.join(args.output_dir, 'trigger_info.th'))

    ## Step 4: Train the Backdoor or Benign Models
    best_poison_acc = 0 
    best_clean_acc = 0
    for epoch in range(1, args.epoch):
        start = time.time()
        lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc = train(model=net, criterion=criterion, optimizer=optimizer,
                                        data_loader=poison_train_loader)

        cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
        po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
        scheduler(epoch)
        end = time.time()
        logger.info(
            '%d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
            epoch, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
            cl_test_loss, cl_test_acc)

        ## Save after couple of epochs
        if (epoch + 1) % args.save_every == 0:
            torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_{}_{}_{}.th'.format(epoch, args.poison_rate, args.rho)))

        elif po_test_acc>=best_poison_acc and cl_test_acc>=best_clean_acc:
            best_poison_acc = po_test_acc
            best_clean_acc = cl_test_acc
            torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_{}_{}_{}.th'.format(args.poison_type, args.poison_rate, args.rho)))

    # Save the last checkpoint
    torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_last_' + str(args.poison_rate) + '_' + str(args.rho) + '.th'))


def train(model, criterion, optimizer, data_loader):
    model.train()
    total_correct = 0
    total_loss = 0.0
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        # optimizer.zero_grad()

        # loss, outputs = criterion_reg.forward_backward_update(inputs, targets, batch_idx)
        # train_loss  += loss.item()
        # _, predicted = outputs.max(1)
        # total   += targets.size(0)
        # correct += predicted.eq(targets).sum().item()
        
        # first forward-backward step
        enable_running_stats(model)
        predictions = model(images)
        loss = smooth_crossentropy(predictions, labels)
        loss.mean().backward()
        optimizer.first_step(zero_grad=True)

        # second forward-backward step
        disable_running_stats(model)
        smooth_crossentropy(model(images), labels).mean().backward()
        optimizer.second_step(zero_grad=True)

        pred = predictions.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.mean().item()


    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc



def test(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


if __name__ == '__main__':
    main()
          
           
           
