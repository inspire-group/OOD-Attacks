import os 
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

import torch
import torch.nn as nn
import numpy as np
import time
import argparse
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

from utils.cifar10_resnet import WideResNet
from utils.train_utils import train_one_epoch, robust_train_one_epoch, update_hyparam
from utils.test_utils import test, robust_test, robust_test_during_train
from utils.data_utils import load_dataset, load_ood_dataset
from utils.io_utils import init_dirs
from utils.ood_utils import ood_trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data args
    parser.add_argument('--dataset_in', type=str, default='CIFAR-10')
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--num_samples', type=int, default=None)

    # Model args
    parser.add_argument('--model', type=str, default='wrn', choices=['wrn'])
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--width', type=int, default=10)

    # Training args
    parser.add_argument('--batch_size', type=int, default=128) 
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--train_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=2e-4)

    # Attack args
    parser.add_argument('--is_adv', dest='is_adv', action='store_true')
    parser.add_argument('--attack', type=str, default='PGD_linf')
    parser.add_argument('--epsilon', type=float, default=0.3)
    parser.add_argument('--attack_iter', type=int, default=10)
    parser.add_argument('--eps_step', type=float, default=0.04)
    parser.add_argument('--targeted', dest='targeted', action='store_true')
    parser.add_argument('--clip_min', type=float, default=0)
    parser.add_argument('--clip_max', type=float, default=1.0)
    parser.add_argument('--rand_init', dest='rand_init', action='store_true')

    # IO args
    parser.add_argument('--last_epoch', type=int, default=0)
    parser.add_argument('--load_checkpoint', dest='load_checkpoint', action='store_true')
    parser.add_argument('--checkpoint_path', type=str, default='trained_models')

    # OOD args
    parser.add_argument('--is_ood_training', dest='ood_training', action='store_true')
    parser.add_argument('--ood_detector', type=str, default=None, choices=['detector_1'])

    if torch.cuda.is_available():
        print('CUDA enabled')
    else:
        raise ValueError('Needs a working GPU!')

    args = parser.parse_args()
    model_dir_name, log_dir_name = init_dirs(args)
    writer = SummaryWriter(log_dir=log_dir_name)
    print('Training %s' % model_dir_name)

    loader_train, loader_test, _ = load_dataset(args, data_dir='./data')

    if args.dataset_in == 'MNIST':
        if 'cnn_3l' in args.model:
            net = cnn_3l()
    elif args.dataset_in == 'CIFAR-10':
        args.epsilon /= 255
        args.eps_step /= 255
        if 'wrn' in args.model:
            net = WideResNet(depth=args.depth, num_classes=args.n_classes, widen_factor=args.width)
    
    if torch.cuda.device_count() > 1:
        print("Using multiple GPUs")
        net = nn.DataParallel(net)
    
    args.batch_size = args.batch_size * torch.cuda.device_count()
    print("Using batch size of {}".format(args.batch_size))

    net.cuda()

    if args.dataset_in == 'MNIST':
        summary(net, (1,28,28))
    
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
        print('Current learning rate: {}'.format(lr))
        if not args.is_adv:
            curr_loss = train_one_epoch(net, criterion, optimizer, 
                                  loader_train, verbose=False)
        else:
            curr_loss = robust_train_one_epoch(net, criterion, optimizer, loader_train, args, verbose=False)
        print('time_taken for #{} epoch = {:.3f}'.format(epoch+1, time.time()-start_time))
        robust_test_during_train(net, loader_test, args, n_batches=10)
        ckpt_path = 'checkpoint_' + str(args.last_epoch)
        torch.save(net.state_dict(), model_dir_name + ckpt_path)
        writer.add_scalar('Loss/train', curr_loss, epoch)
        writer.add_scalar('Lr', lr, epoch)