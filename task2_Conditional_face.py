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

# def interpolations(z1, z2, n):
# 	# print('interpolations input size',z1.size())
# 	z_list = torch.Tensor([]).cuda()
# 	for j in range(n):
# 		top = z1
# 		down = z2
# 		value = down + 1.0 * j*(top-down)/n
# 		z_list = torch.cat((z_list,value.unsqueeze(0)),0)
# 		# print('interpolations output size', z_list.size())
# 	return z_list


if __name__ == "__main__":
	device = torch.device("cuda")

	output_folder = '0614_0641_logs/'
	model_name = 'glow_checkpoint_75000.pt'

	with open(output_folder + 'hparams.json') as json_file:
		hparams = json.load(json_file)

	image_shape = (64,64,3)
	num_classes = 40
	model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
				 hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes,
				 hparams['learn_top'], hparams['y_condition'])

	model.load_state_dict(torch.load(output_folder + model_name, map_location="cpu")['model'])
	model.set_actnorm_init()
	model = model.to(device)
	model = model.eval()


	attribute_list = [20, 31, 26, 16] # Male, Smiling, Pale_Skin, Goatee
	generate_x_list = torch.Tensor([]).cuda()
	for yes in [1,0]:
		for i,attribute in enumerate(attribute_list):
			z = torch.rand( (1, 48, 8, 8) )
			y = torch.zeros(40).unsqueeze(dim=1)
			y[attribute] = yes
			predict_x = model(y_onehot=y.cuda(), z=z.cuda(), temperature=1, reverse=True)
			generate_x_list = torch.cat((generate_x_list,predict_x), 0)
	save_image(generate_x_list, 'images/task2_Conditional_face.png')
