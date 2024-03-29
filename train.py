from __future__ import print_function
import argparse
import torch
from torch.utils.data import DataLoader
from train_loop import TrainLoop
import torch.optim as optim
from torchvision import datasets, transforms
from models import vgg, resnet, densenet
import numpy as np
from time import sleep
import os
import sys

from utils import *

# Training settings
parser = argparse.ArgumentParser(description='Cifar10 Classification')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--valid-batch-size', type=int, default=256, metavar='N', help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 500)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--l2', type=float, default=5e-4, metavar='lambda', help='L2 wheight decay coefficient (default: 0.0005)')
parser.add_argument('--patience', type=int, default=10, metavar='S', help='Epochs to wait before decreasing LR by a factor of 0.5 (default: 10)')
parser.add_argument('--slack', type=float, default=1.1, metavar='l', help='nadir point multiplier, should always be greater than 1 (default: 1.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='lambda', help='Momentum (default: 0.9)')
parser.add_argument('--train-mode', choices=['vanilla', 'hyper'], default='vanilla', help='Salect train mode. Default is vanilla')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--data-path', type=str, default='./data/cifar10_train_data.hdf', metavar='Path', help='Path to data')
parser.add_argument('--valid-data-path', type=str, default='./data/cifar10_test_data.hdf', metavar='Path', help='Path to data')
parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')
parser.add_argument('--n-workers', type=int, default=4, metavar='N', help='Workers for data loading. Default is 4')
parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet'], default='resnet')
parser.add_argument('--save-every', type=int, default=1, metavar='N', help='how many epochs to wait before logging training status. Default is 1')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
parser.add_argument('--no-cp', action='store_true', default=False, help='Disables checkpointing')
parser.add_argument('--verbose', type=int, default=1, metavar='N', help='Verbose is activated if > 0')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

if args.cuda:
	torch.backends.cudnn.benchmark=True

transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([x / 255 for x in [125.3, 123.0, 113.9]], [x / 255 for x in [63.0, 62.1, 66.7]])])
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize([x / 255 for x in [125.3, 123.0, 113.9]], [x / 255 for x in [63.0, 62.1, 66.7]])])

#trainset = Loader(args.data_path)
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, worker_init_fn=set_np_randomseed)

#validset = Loader(args.valid_data_path)
validset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.n_workers)

if args.model == 'vgg':
	model = vgg.VGG('VGG16')
elif args.model == 'resnet':
	model = resnet.ResNet50()
elif args.model == 'densenet':
	model = densenet.densenet_cifar()

if args.cuda:
	device = get_freer_gpu()
	model = model.cuda(device)

optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.l2, momentum=args.momentum)

trainer = TrainLoop(model, optimizer, train_loader, valid_loader, slack=args.slack, train_mode=args.train_mode, patience=args.patience, verbose=args.verbose, save_cp=(not args.no_cp), checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, cuda=args.cuda)

if args.verbose >0:
	print('Cuda Mode is: {}'.format(args.cuda))
	print('Selected model: {}'.format(args.model))
	print('Train mode: {}'.format(args.train_mode))
	print('Batch size: {}'.format(args.batch_size))
	print('LR: {}'.format(args.lr))
	print('Momentum: {}'.format(args.momentum))
	print('l2: {}'.format(args.l2))
	print('slack: {}'.format(args.slack))
	print('Patience: {}'.format(args.patience))

trainer.train(n_epochs=args.epochs, save_every=args.save_every)
