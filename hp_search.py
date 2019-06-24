import nevergrad.optimization as optimization
from nevergrad import instrumentation as instru
import argparse
import torch
from torch.utils.data import DataLoader
from train_loop import TrainLoop
import torch.optim as optim
from torchvision import datasets, transforms
from models import vgg, resnet, densenet
import numpy as np
import os 
import sys

from utils import *

def get_cp_name(dir_):

	idx = np.random.randint(1)

	fname = dir_ + '/' + str(np.random.randint(1,999999999,1)[0]) + '.pt'

	while os.path.isfile(fname):
		fname = dir_ + str(np.random.randint(1,999999999,1)[0]) + '.pt'

	return fname.split('/')[-1]

# Training settings
parser = argparse.ArgumentParser(description='Cifar10 Classification')
parser.add_argument('--batch-size', type=int, default=24, metavar='N', help='input batch size for training (default: 24)')
parser.add_argument('--valid-batch-size', type=int, default=16, metavar='N', help='input batch size for testing (default: 16)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--data-path', type=str, default='./data/cifar10_train_data.hdf', metavar='Path', help='Path to data')
parser.add_argument('--valid-data-path', type=str, default='./data/cifar10_test_data.hdf', metavar='Path', help='Path to data')
parser.add_argument('--n-workers', type=int, default=4, metavar='N', help='Workers for data loading. Default is 4')
parser.add_argument('--budget', type=int, default=100, metavar='N', help='Maximum training runs')
parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet', 'all'], default='resnet')
parser.add_argument('--train-mode', choices=['vanilla', 'hyper', 'all'], default='all', help='Salect train mode. Default is vanilla')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

def train(lr, l2, momentum, slack, patience, model, epochs, batch_size, valid_batch_size, train_mode, n_workers, cuda, data_path, valid_data_path, checkpoint_path):

	cp_name = get_cp_name(checkpoint_path)

	transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([x / 255 for x in [125.3, 123.0, 113.9]], [x / 255 for x in [63.0, 62.1, 66.7]])])
	transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize([x / 255 for x in [125.3, 123.0, 113.9]], [x / 255 for x in [63.0, 62.1, 66.7]])])

	#trainset = Loader(data_path)
	trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
	train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=n_workers, worker_init_fn=set_np_randomseed)

	#validset = Loader(valid_data_path)
	validset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
	valid_loader = torch.utils.data.DataLoader(validset, batch_size=valid_batch_size, shuffle=False, num_workers=n_workers)

	if model == 'vgg':
		model_ = vgg.VGG('VGG16')
	elif model == 'resnet':
		model_ = resnet.ResNet50()
	elif model == 'densenet':
		model_ = densenet.densenet_cifar()

	if cuda:
		device = get_freer_gpu()
		model_ = model_.cuda(device)

	optimizer = optim.SGD(model_.parameters(), lr=lr, weight_decay=l2, momentum=momentum)

	trainer = TrainLoop(model_, optimizer, train_loader, valid_loader, slack=slack, train_mode=train_mode, patience=int(patience), verbose=-1, cp_name=cp_name, save_cp=True, checkpoint_path=checkpoint_path, cuda=cuda)

	for i in range(5):

		if i>0:
			print(' ')
			print('Trial {}'.format(i+1))
			print(' ')

		try:
			cost = trainer.train(n_epochs=epochs, save_every=epochs+10)

			print(' ')
			print('Best cost in file ' + cp_name + 'was: {}'.format(cost))
			print(' ')
			print('With hyperparameters:')
			print('Selected model: {}'.format(model))
			print('Train mode: {}'.format(train_mode))
			print('Batch size: {}'.format(batch_size))
			print('LR: {}'.format(lr))
			print('Momentum: {}'.format(momentum))
			print('l2: {}'.format(l2))
			print('Slack: {}'.format(slack))
			print('Patience: {}'.format(patience))
			print(' ')

			return cost
		except:
			pass

	print('Returning dummy cost due to failures while training.')
	print('With hyperparameters:')
	print('Selected model: {}'.format(model))
	print('Train mode: {}'.format(train_mode))
	print('Batch size: {}'.format(batch_size))
	print('LR: {}'.format(lr))
	print('Momentum: {}'.format(momentum))
	print('l2: {}'.format(l2))
	print('Slack: {}'.format(slack))
	print('Patience: {}'.format(patience))
	print(' ')
	return 0.99

lr = instru.var.Array(1).asfloat().bounded(1, 4).exponentiated(base=10, coeff=-1)
l2 = instru.var.Array(1).asfloat().bounded(1, 5).exponentiated(base=10, coeff=-1)
momentum = instru.var.Array(1).asfloat().bounded(0.01, 0.99)
slack = instru.var.Array(1).asfloat().bounded(1.01, 3.00)
patience = instru.var.Array(1).asfloat().bounded(1, 100)
model=instru.var.OrderedDiscrete(['vgg', 'resnet', 'densenet']) if args.model=='all' else args.model
epochs = args.epochs
batch_size = args.batch_size
valid_batch_size = args.valid_batch_size
train_mode=instru.var.OrderedDiscrete(['vanilla', 'hyper']) if args.train_mode=='all' else args.train_mode
n_workers = args.n_workers
cuda = args.cuda
data_path = args.data_path
valid_data_path = args.valid_data_path
checkpoint_path=args.checkpoint_path

instrum = instru.Instrumentation(lr, l2, momentum, slack, patience, model, epochs, batch_size, valid_batch_size, train_mode, n_workers, cuda, data_path, valid_data_path, checkpoint_path)

hp_optimizer = optimization.optimizerlib.RandomSearch(instrumentation=instrum, budget=args.budget)

print(hp_optimizer.optimize(train, verbosity=2))
