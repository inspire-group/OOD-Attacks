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
from utils.test_utils import test, robust_test
from utils.data_utils import load_cifar_dataset
from utils.io_utils import init_dirs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_in', type=str, default='CIFAR-10')
    parser.add_argument('--model', type=str, default='wrn', choices=['wrn'])
    # parser.add_argument('--method', default="mixtrain")
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--width', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128) 
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--is_adv', type=bool, default=False)
    # parser.add_argument('--learning_rate', type=float, default=0.1)
    # parser.add_argument('--weight_decay', type=float, default=2e-4)
    parser.add_argument('--attack', type=str, default='PGD')
    parser.add_argument('--epsilon', type=float, default=8.0)
    parser.add_argument('--attack_iter', type=int, default=10)
    parser.add_argument('--eps_step', type=float, default=2.0)
    parser.add_argument('--targeted', type=bool, default=False)
    parser.add_argument('--clip_min', type=float, default=0)
    parser.add_argument('--clip_max', type=float, default=1.0)
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--last_epoch', type=int, default=0)
    parser.add_argument('--checkpoint_path', type=str, default='trained_models')
    parser.add_argument('--is_ood', type=bool, default=False)
    parser.add_argument('--dataset_out', type=str, default='Imagenet', choices=['mnist', 'cifar', 'imagenet', 'voc12',
                      'gaussian_noise', 'uniform_noise', 'random_photograph', 'ones', 'zeros'])
    

    args = parser.parse_args()
    model_dir_name = init_dirs(args)
    loader_train, loader_test = load_dataset(args, data_dir='./data')

    if args.dataset == 'CIFAR-10':
        args.epsilon /= 255
        args.eps_step /= 255
   
    if 'wrn' in args.model:
        net = WideResNet(depth=args.depth, num_classes=args.n_classes, widen_factor=args.width)
    else:
        raise ValueError('This type of model is not defined')
    
    if torch.cuda.device_count() > 1:
        print("Using multiple GPUs")
        net = nn.DataParallel(net)
    args.batch_size = args.batch_size * torch.cuda.device_count()
    print("Using batch size of {}".format(args.batch_size))
    if torch.cuda.is_available():
        print('CUDA enabled.')
        net.cuda()

    criterion = nn.CrossEntropyLoss()  

    net.eval()
    ckpt_path = 'checkpoint_' + str(args.last_epoch)
    net.load_state_dict(torch.load(model_dir_name + ckpt_path))
    robust_test(net, loader_test, args, n_batches=10)