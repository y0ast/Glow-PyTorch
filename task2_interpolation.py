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
    z_list = []
    for j in range(n):
        list_n = []
        for i in range(len(z1)):
            top = z1[i]
            down = z2[i]
            value = down + 1.0 * j*(top-down)/n
            list_n.append(value)
        z_list.append(list_n)
    return np.array(z_list)


if __name__ == "__main__":
    device = torch.device("cuda")

    output_folder = '0614_0641_logs/'
    model_name = 'glow_checkpoint_75000.pt'

    with open(output_folder + 'hparams.json') as json_file:
        hparams = json.load(json_file)

    image_shape = (64,64,3)
    num_classes = 40
    dataset_test = CelebALoader(root_folder=hparams['dataroot']) #'/home/yellow/deep-learning-and-practice/hw7/dataset/task_2/'
    test_loader = DataLoader(dataset_test,batch_size=1,shuffle=False,drop_last=True)
    model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
                 hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes,
                 hparams['learn_top'], hparams['y_condition'])

    model.load_state_dict(torch.load(output_folder + model_name, map_location="cpu")['model'])
    model.set_actnorm_init()
    model = model.to(device)
    model = model.eval()

    face_pair = [(1,3), (11,30), (23,27)]
    N = 8
    interpolation_result = []

    for p in range(len(face_pair)):
        x1, y1, x2, y2 = None, None, None, None
        for i,(x,y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)
            if i == face_pair[p][0] :
                x1, y1 = x, y
            if i == face_pair[p][1] :
                x2, y2 = x, y
                break

        z1, bpd, y_logits = model(x1, y1)  # return: z, bpd, y_logits
        z2, bpd, y_logits = model(x2, y2)  # return: z, bpd, y_logits
        z1 = z1.cpu()
        z2 = z2.cpu()
        z_list = interpolations(z1[0].detach().numpy(), z2[0].detach().numpy(), N)
        z_list = torch.Tensor(z_list).cuda()
        y_rand = torch.rand(40).unsqueeze(dim=1)
        predict_x = model(y_onehot=y_rand, z=z_list, temperature=1, reverse=True)
        print(predict_x.size())
        for k in range(len(predict_x)): interpolation_result.append(predict_x[k])


    save_image(interpolation_result, 'task2_interpolation.png', normalize=True)
