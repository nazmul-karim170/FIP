import os
import time
import argparse
import numpy as np
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import copy
import math
import networks
import torch.nn.functional as F
import pandas as pd
import data.badnets_blend as poison
from torch.autograd import Variable
from PIL import Image
from data.dataloader_cifar import *
import matplotlib.pyplot as plt
import random 
from Regularizer import CDA_Regularizer as regularizer   ## Regularizer 
import torch.autograd as AG
import logging


## Basic Model Parameters.
parser = argparse.ArgumentParser(description='Train poisoned networks')

parser.add_argument('--arch', type=str, default='resnet18',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'MobileNetV2', 'vgg19_bn'])
parser.add_argument('--widen-factor', type=int, default=1, help='Widen_Factor for WideResNet')
parser.add_argument('--batch-size', type=int, default=128, help='the batch size for dataloader')
parser.add_argument('--epoch',      type = int, default = 250, help='the numbe of epoch for training')
parser.add_argument('--schedule',   type=int, nargs='+', default=[100, 150], help='Decrease learning rate at these epochs.')
parser.add_argument('--save-every', type=int, default=20, help='save checkpoints every few epochs')
parser.add_argument('--data-dir',   type=str, default='../data', help='dir to the dataset')
parser.add_argument('--output-dir', type=str, default='logs/models/Spectral_smooth/')
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
parser.add_argument("--reg_F", default=0.5, type=float, help="CDA Regularizer Coefficient, eta_F")


args = parser.parse_args()
args_dict = vars(args)
# os.makedirs(args.output_dir, exist_ok=True)
random.seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", device)
# torch.cuda.set_device(args.gpuid)

def main(transform_train, transform_test):

    ## Step 0: Data Transformation 
    logger = logging.getLogger(__name__)
    args.output_dir = os.path.join(args.output_dir, "output_" + str(args.poison_rate) + "_" + str(args.reg_F) )
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = "output_" + str(args.poison_rate) + "_" + str(args.reg_F) + '.log'
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


    ## Step 2: Load Model Checkpoints
    # state_dict = torch.load(args.checkpoint, map_location=device)
    if args.poison_type in ['Dynamic']:
        state_dict = torch.load(args.checkpoint, map_location=device)['netC']

    net = getattr(networks, args.arch)(num_classes=10)                ## For Mask-finetuning 
    
    ## Step 2: Load model checkpoints 
    # net.load_state_dict(state_dict)
    net = net.cuda()
    net.train()


    ## Step 3: Training Settings
    criterion = torch.nn.CrossEntropyLoss().cuda()
    nb_iterations = int(np.ceil(args.epoch / 1))

    ## Initialize FIM
    criterion_reg = regularizer(args, device, net, criterion, args.epoch)
    # criterion_reg.register_ewc_params(clean_val, 100, 100)   ## Store the gradient information and FIM (we calculate FIM only once)


    ## Step 3: train backdoored models
    # N_c = len(clean_val)/args.num_classes  


    ## Step 4: Validate the Given Model 
    cl_test_loss, ACC = FIP_Test(model=net, criterion=criterion, data_loader=clean_test_loader)
    po_test_loss, ASR = FIP_Test(model=net, criterion=criterion, data_loader=poison_test_loader)
    print("ASR and ACC Before Purification\t")
    print('-----------------------------------------------------------------')
    print('ASR \t ACC')
    print('{:.4f} \t {:.4f}'.format(100*ASR, 100*ACC))
    print('-----------------------------------------------------------------')
    # print("validation Size:", len(clean_val))
    # print("Number of Samples per Class:", N_c)


    ## Losses and Accuracy 
    clean_losses  = np.zeros(nb_iterations)
    poison_losses = np.zeros(nb_iterations)
    clean_accs    = np.zeros(nb_iterations)
    poison_accs   = np.zeros(nb_iterations)

    
    ## Step 5: Purification Process Starts
    print('-----------------------------------------------------------------')
    print('-----------------------------------------------------------------')
    print('-----------------------------------------------------------------')
    print('-----------------------------------------------------------------')
    print("ASR and ACC After Purification\t")
    print('-----------------------------------------------------------------')
    print('Iter \t ASR \t \t ACC')
    for i in range(nb_iterations):
        lr = args.lr
        train_loss, train_acc = FIP_Train(args,i, net, clean_test, poison_train_loader, criterion_reg)

        clean_loss , ACC = FIP_Test(model=net, criterion=criterion, data_loader=clean_test_loader)
        poison_loss, ASR = FIP_Test(model=net, criterion=criterion, data_loader=poison_test_loader)

        clean_losses[i]  = clean_loss
        poison_losses[i] = poison_loss
        clean_accs[i]    = ACC
        poison_accs[i]   = ASR

        ## Save Stattistics and the Purified model
        np.savez(os.path.join(args.output_dir,'remove_model_'+ args.poison_type + '_' + str(args.dataset) + '_.npz'), cl_loss = clean_losses, cl_test = clean_accs, po_loss = poison_losses, po_acc = poison_accs)
        model_save = args.poison_type + '_' + str(i) + '_' + str(args.reg_F) + '_' + str(args.poison_rate) + '.pth'
        torch.save(net.state_dict(), os.path.join(args.output_dir, model_save))
        # scheduler.step()

        print('{} \t {:.4f} \t {:.4f}'.format((i + 1) , 100*ASR, 100*ACC))

## Loading the Pre-trained Weights to the Current Model
def load_model(net, orig_state_dict):
    if 'state_dict' in orig_state_dict.keys():
        orig_state_dict = orig_state_dict['state_dict']
    if "state_dict" in orig_state_dict.keys():
        orig_state_dict = orig_state_dict["state_dict"]

    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
        elif 'running_mean_noisy' in k or 'running_var_noisy' in k or 'num_batches_tracked_noisy' in k:
            new_state_dict[k] = orig_state_dict[k[:-6]].clone().detach()
        else:
            new_state_dict[k] = v

    net.load_state_dict(new_state_dict)


def get_trace_loss(model, loss, params, hi=10):

    niters = hi
    V = list()
    for _ in range(niters):
        # V_i = [torch.randint_like(p, high=2, device=device) for p in model.parameters()]
        # for V_ij in V_i:
        #     V_ij[V_ij == 0] = -1
        V_i = [torch.randn_like(p, device=device) for p in params]
        V.append(V_i)

        ### 
    trace = list()
    grad = AG.grad(loss, params, create_graph=True)

    for V_i in V:
        Hv = AG.grad(grad, params, V_i, create_graph=True)
        this_trace = 0.0
        for Hv_, V_i_ in zip(Hv, V_i):
            this_trace = this_trace + torch.sum(Hv_ * V_i_)
        trace.append(this_trace)

    return sum(trace) / niters

## Training Scheme
def FIP_Train(args,epoch, net, clean_val, clean_val_loader, criterion_reg):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            ('Fisher', args.lr, 0, 0, correct, total))
    prog_bar = tqdm(enumerate(clean_val_loader), total=len(clean_val_loader), desc=desc, leave=True)

    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()

        loss, outputs = criterion_reg.forward_backward_update(inputs, targets, batch_idx)
        train_loss  += loss.item()
        _, predicted = outputs.max(1)
        total   += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                ('Fisher', args.lr, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)


    return train_loss/(batch_idx + 1), 100. * correct / total


def FIP_Test(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.cuda(), torch.squeeze(labels.cuda())
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = torch.max(output,1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Remove Backdoor Through Neural Fine-Tuning')

    # # Basic model parameters.
    # parser.add_argument('--arch', type=str, default='resnet18',
    #                     choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'])
    # parser.add_argument('--checkpoint', type=str, required=True, help='The checkpoint to be pruned')
    # parser.add_argument('--widen-factor', type=int, default=1, help='widen_factor for WideResNet')
    # parser.add_argument('--batch-size', type=int, default=128, help='the batch size for dataloader')       
    # parser.add_argument('--lr', type=float, default=0.005, help='the learning rate for mask optimization')   
    # parser.add_argument('--nb-epochs', type=int, default=1000, help='the number of iterations for training')  
    # parser.add_argument('--epoch-aggregation', type=int, default=250, help='print results every few iterations')  
    # parser.add_argument('--data-dir', type=str, default='../data', help='dir to the dataset')
    # parser.add_argument('--val-ratio', type=float, default=0.01, help='The fraction of the validate set')  ## Controls the validation size
    # parser.add_argument('--output-dir', type=str, default='save/purified_networks/')
    # parser.add_argument('--gpuid', type=int, default=0, help='the transparency of the trigger pattern.')

    # parser.add_argument('--poison-type', type=str, default='badnets', choices=['badnets', 'Feature', 'FC',  'SIG', 'Dynamic', 'TrojanNet', 'blend', 'CLB', 'benign'],
    #                     help='type of backdoor attacks used during training')
    # parser.add_argument('--trigger-alpha', type=float, default=0.2, help='the transparency of the trigger pattern.')

    # parser.add_argument('--log_root', type=str, default='./logs', help='logs are saved here')
    # parser.add_argument('--dataset', type=str, default='CIFAR10', help='name of image dataset')
    # parser.add_argument('--load_fixed_data', type=int, default=1, help='load the local poisoned test dataest')
    # parser.add_argument('--poisoned_data_test_all2one', type=str, default='./data/dynamic/poisoned_data/cifar10-test-inject0.1-target0-dynamic-all2one.npy', help='random seed')
    # parser.add_argument('--poisoned_data_test_all2all', type=str, default='./data/dynamic/poisoned_data/cifar10-test-inject0.1-target0-dynamic-all2all_mask.npy', help='random seed')

    # parser.add_argument('--TCov', default=10, type=int)                   ## 10 works fine 
    # parser.add_argument('--target_label', type=int, default=0, help='class of target label')
    # parser.add_argument('--trigger_type', type=str, default='squareTrigger', choices=['squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger',
    #                                'signalTrigger', 'trojanTrigger'], help='type of backdoor trigger')
    # parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
    # parser.add_argument('--trig_w', type=int, default=1, help='width of trigger pattern')
    # parser.add_argument('--trig_h', type=int, default=1, help='height of trigger pattern')    
    # parser.add_argument('--alpha', type=float, default=0.8, help='Search area design Parameter')
    # parser.add_argument('--beta', type=float, default=0.5, help='Search area design Parameter')
    # parser.add_argument('--num_classes', type=float, default=10, help='Number of classes')

    # Linear Transformation
    MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
    STD_CIFAR10  = (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])

    main(transform_train, transform_test)
