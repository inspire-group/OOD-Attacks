import os 
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
import numpy as np
import time
import argparse

from utils.cifar10_models import WideResNet
from utils.train_utils import train_one_epoch, robust_train_one_epoch, update_hyparam
from utils.test_utils import test, robust_test
from utils.data_utils import load_cifar_dataset
from utils.io_utils import init_dirs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR-10')
    parser.add_argument('--model', type=str, default='wrn', choices=['wrn'])
    # parser.add_argument('--method', default="mixtrain")
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--width', type=int, default=1)
    parser.add_argument('--is_training', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=128) 
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--train_epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=2e-4)
    parser.add_argument('--is_adv', type=bool, default=False)
    parser.add_argument('--attack', type=str, default='PGD')
    parser.add_argument('--epsilon', type=float, default=8.0)
    parser.add_argument('--attack_iter', type=int, default=10)
    parser.add_argument('--eps_step', type=float, default=2.0)
    parser.add_argument('--targeted', type=bool, default=False)
    parser.add_argument('--clip_min', type=float, default=0)
    parser.add_argument('--clip_max', type=float, default=1.0)
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--last_epoch', type=int, default=0)
    parser.add_argument('--load_checkpoint', type=bool, default=False)
    parser.add_argument('--checkpoint_path', type=str, default='trained_models')
    

    if torch.cuda.is_available():
        print('CUDA enabled')
    else:
        raise ValueError('Needs a working GPU!')

    args = parser.parse_args()
    model_dir_name = init_dirs(args)
    if args.dataset == 'CIFAR-10':
        loader_train, loader_test = load_cifar_dataset(args, data_dir='./data')

    if args.dataset == 'CIFAR-10':
        args.epsilon /= 255
        args.eps_step /= 255
   
    net = WideResNet(depth=args.depth, num_classes=args.n_classes, widen_factor=args.width)
    if torch.cuda.device_count() > 1:
        print("Using multiple GPUs")
        net = nn.DataParallel(net)
    
    args.batch_size = args.batch_size * torch.cuda.device_count()
    print("Using batch size of {}".format(args.batch_size))

    net.cuda()
    
    if args.load_checkpoint:
        net.load_state_dict(torch.load(model_dir_name))
        robust_test(net, loader_test, args, n_batches=10)

    criterion = nn.CrossEntropyLoss()  

    optimizer = torch.optim.SGD(net.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    for epoch in range(args.last_epoch, args.train_epochs):
        start_time = time.time()
        lr = update_hyparam(epoch, args)
        optimizer.param_groups[0]['lr'] = lr
        print('Current learning rate: %s'.format(lr))
        if not args.is_adv:
            train_one_epoch(net, criterion, optimizer, 
                                  loader_train, verbose=False)
        if args.is_adv:
            robust_train_one_epoch(net, criterion, optimizer, loader_train, args, targeted=args.targeted, verbose=False)
        print('time_taken for #{} epoch = {:.3f}'.format(epoch+1, time.time()-start_time))
        robust_test(net, loader_test, args, n_batches=10)
        ckpt_path = 'checkpoint_' + str(args.last_epoch)
        torch.save(net.state_dict(), model_dir_name + ckpt_path)