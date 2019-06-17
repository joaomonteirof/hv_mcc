from __future__ import print_function
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
from models import vgg, resnet, densenet

# Training settings
parser = argparse.ArgumentParser(description='Test new architectures')
parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet', 'all'], default='resnet')
parser.add_argument('--hidden-size', type=int, default=512, metavar='S', help='latent layer dimension (default: 512)')
parser.add_argument('--n-hidden', type=int, default=1, metavar='N', help='maximum number of frames per utterance (default: 1)')
args = parser.parse_args()

if args.model == 'vgg' or args.model =='all':
	model = vgg.VGG('VGG16')
	batch = torch.rand(3, 3, 32, 32)
	out = model.forward(batch)
	pred = model.out_proj(out)
	print('vgg', out.size(), pred.size())
if args.model == 'resnet' or args.model =='all':
	model = resnet.ResNet50()
	batch = torch.rand(3, 3, 32, 32)
	out = model.forward(batch)
	print('resnet', out.size(), pred.size())
if args.model == 'densenet' or args.model =='all':
	model = densenet.densenet_cifar()
	batch = torch.rand(3, 3, 32, 32)
	out = model.forward(batch)
	print('densenet', out.size(), pred.size())


