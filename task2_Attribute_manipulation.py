import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from datasets import get_CIFAR10, get_SVHN
from model import Glow

import argparse
import os
import json
import shutil
import random
from itertools import islice

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage, Loss

from datasets import get_CIFAR10, get_SVHN
from mydataset import get_CelebA_data, CelebALoader
from model import Glow
from utils import save_image

def interpolations(z1, z2, n):
	# print('interpolations input size',z1.size())
	z_list = torch.Tensor([]).cuda()
	for j in range(n):
		top = z1
		down = z2
		value = down + 1.0 * j*(top-down)/n
		z_list = torch.cat((z_list,value.unsqueeze(0)),0)
		# print('interpolations output size', z_list.size())
	return z_list


if __name__ == "__main__":
	device = torch.device("cuda")

	output_folder = '0614_0641_logs/'
	model_name = 'glow_checkpoint_75000.pt'

	with open(output_folder + 'hparams.json') as json_file:
		hparams = json.load(json_file)

	image_shape = (64,64,3)
	num_classes = 40
	Batch_Size = 4
	dataset_test = CelebALoader(root_folder=hparams['dataroot']) #'/home/yellow/deep-learning-and-practice/hw7/dataset/task_2/'
	test_loader = DataLoader(dataset_test,batch_size=Batch_Size,shuffle=False,drop_last=True)
	model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
				 hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes,
				 hparams['learn_top'], hparams['y_condition'])

	model.load_state_dict(torch.load(output_folder + model_name, map_location="cpu")['model'])
	model.set_actnorm_init()
	model = model.to(device)
	model = model.eval()

	# attribute_list = [8] # Black_Hair
	attribute_list = [20, 31, 26] # Male, Smiling, Pale_Skin
	# attribute_list = [11, 26, 31, 8, 6, 7] # Brown_Hair, Pale_Skin, Smiling, Black_Hair, Big_Lips, Big_Nose
	# attribute_list = [i for i in range(40)]
	N = 8
	z_pos_list = [ torch.Tensor([]).cuda() for i in range(len(attribute_list)) ]
	z_neg_list = [ torch.Tensor([]).cuda() for i in range(len(attribute_list)) ]

	z_input_img = None
	with torch.no_grad():
		for i,(x,y) in enumerate(test_loader):
			print('reading data: ',i,'/',30000/Batch_Size)
			# if i >= 10: break
			for j,attribute_num in enumerate(attribute_list):
				x = x.to(device)
				y = y.to(device)
				z, bpd, y_logits = model(x, y_onehot=y)  # return: z, bpd, y_logits

				for k in range(len(z)):
					if  y[k][attribute_num] == 1:
						z_pos_list[j] = torch.cat((z_pos_list[j],z[k].unsqueeze(0)),0)
					else:
						z_neg_list[j] = torch.cat((z_neg_list[j],z[k].unsqueeze(0)),0)
		### z_pos_list size: ( N_attribute * N_pos * 48 * 8 * 8 )

		z_pos_mean = torch.Tensor([]).cuda()
		z_neg_mean = torch.Tensor([]).cuda()
		for i in range(len(attribute_list)):
			pos_mean = torch.mean(z_pos_list[j], 0) #pos_mean size: torch.Size([48, 8, 8])
			z_pos_mean = torch.cat((z_pos_mean, pos_mean.unsqueeze(0)), 0)
			neg_mean = torch.mean(z_neg_list[j], 0)
			z_neg_mean = torch.cat((z_neg_mean, neg_mean.unsqueeze(0)), 0)
		### z_pos_mean size: torch.Size([N_attribute, 48, 8, 8])

		z_input_img = z[0].clone()
		y_input_img = y[0].clone()
		y_rand = torch.rand(40).cuda()
		generate_x_list = torch.Tensor([]).cuda()
		for i,attribute_num in enumerate(attribute_list):
			if y_input_img[attribute_num] == 1:
				inters = interpolations(z_input_img, z_neg_mean[i], n=N)
			else:
				inters = interpolations(z_pos_mean[i], z_input_img, n=N)
			### inters size: torch.Size([N, 48, 8, 8]), N=num of interpolations

			for n in range(N):
				z_for_generate = inters[n].unsqueeze(0)
				y_for_generate = y_input_img.clone().unsqueeze(0)
				predict_x = model(y_onehot=y_for_generate, z=z_for_generate, temperature=1, reverse=True)
				generate_x_list = torch.cat((generate_x_list,predict_x), 0)
				print('generate_x_list',generate_x_list.size())

		### generate_x_list size: torch.Size([N_attribute*N_interpolation, 3, 64, 64])


		save_image(generate_x_list, 'images/task2_Attribute_manipulation.png', normalize=True)
