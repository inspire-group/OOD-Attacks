import os 
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
import numpy as np
import time
import argparse

from utils.cifar10_models import WideResNet
from utils.train_test import train_one_epoch, test, madry_train_one_epoch, test_madry

def load_cifar_dataset(args, data_dir):
    # CIFAR-10 data loaders
    trainset = datasets.CIFAR10(root=data_dir, train=True,
                                download=True, 
                                transform=transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, 4),
                                    transforms.ToTensor()
                                ]))
    loader_train = torch.utils.data.DataLoader(trainset, 
                                batch_size=args.batch_size,
                                shuffle=True)

    testset = datasets.CIFAR10(root=data_dir,
                                train=False,
                                download=True, transform=transforms.ToTensor())
    loader_test = torch.utils.data.DataLoader(testset, 
                                batch_size=args.test_batch_size,
                                shuffle=False)
    return loader_train, loader_test


def update_hyperparameters(epoch, args):
    #args.learning_rate = args.learning_rate * (0.6 ** ((max((epoch-args.schedule_length), 0) // 5)))
    lr_steps = [100, 150, 200]
    lr = args.learning_rate
    for i in lr_steps:
        if epoch<i:
            break
        lr /= 10
    return lr 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default="mixtrain")
    parser.add_argument('--is_training', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=128) 
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--train_epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=2e-4)
    parser.add_argument('--epsilon', type=float, default=8.0/255.0)
    parser.add_argument('--attack_iter', type=int, default=10)
    parser.add_argument('--eps_step', type=float, default=2.0/255)
    parser.add_argument('--targeted', type=bool, default=False)
    parser.add_argument('--clip_min', type=float, default=0)
    parser.add_argument('--clip_max', type=float, default=1.0)
    parser.add_argument('--model_path', type=str, default='./trained_models/')
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--width', type=int, default=10)
    parser.add_argument('--is_adv', type=bool, default=False)
    parser.add_argument('--last_epoch', type=int, default=0)
    parser.add_argument('--load_checkpoint', type=bool, default=False)
    parser.add_argument('--checkpoint_path', type=str, default='./trained_models/Robust_wrn_28_with_std_10.pth')
    

    args = parser.parse_args()
    loader_train, loader_test = load_cifar_dataset(args, data_dir='./data')
   
    net = WideResNet(depth=28, num_classes=args.n_classes, widen_factor=args.width)
    if torch.cuda.device_count() > 1:
        print("Using multiple GPUs")
        net = nn.DataParallel(net)
    args.batch_size = args.batch_size * torch.cuda.device_count()
    print("Using batch size of {}".format(args.batch_size))
    if torch.cuda.is_available():
        print('CUDA enabled.')
        net.cuda()
    
    if args.load_checkpoint:
        net.load_state_dict(torch.load(args.checkpoint_path))
        test_madry(net, loader_test, args, n_steps=10)

    
    criterion = nn.CrossEntropyLoss()  
    if args.is_training:
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=args.learning_rate,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay)
        for epoch in range(args.last_epoch, args.train_epochs):
            start_time = time.time()
            lr = update_hyperparameters(epoch, args)
            optimizer.param_groups[0]['lr'] = lr
            print(lr)
            if not args.is_adv:
                train_one_epoch(net, criterion, optimizer, 
                                      loader_train, verbose=False)
            if args.is_adv:
                madry_train_one_epoch(net, criterion, optimizer, loader_train, args, targeted=args.targeted, verbose=False)
            print('time_taken for #{} epoch = {:.3f}'.format(epoch+1, time.time()-start_time))
            test_madry(net, loader_test, args, n_steps=10)
            torch.save(net.state_dict(), './trained_models/robust_wrn_28_with_std_{}_checkpoint_{}.pth'.format(args.width, args.last_epoch))
    else:
        net.eval()
        net.load_state_dict(torch.load('./trained_models/robust_wrn_28_with_std_{}_checkpoint_{}.pth'.format(args.width, args.last_epoch)))