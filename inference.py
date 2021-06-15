import json

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


if __name__ == "__main__":
    device = torch.device("cuda")

    output_folder = '0614_0641_logs/'
    model_name = 'glow_checkpoint_75000.pt'

    with open(output_folder + 'hparams.json') as json_file:
        hparams = json.load(json_file)

    image_shape = (64,64,3)
    num_classes = 40
    dataset_test = CelebALoader(root_folder=hparams['dataroot']) #'/home/yellow/deep-learning-and-practice/hw7/dataset/task_2/'
    test_loader = DataLoader(dataset_test,batch_size=32,shuffle=False,drop_last=True)
    model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
                 hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes,
                 hparams['learn_top'], hparams['y_condition'])


    model.load_state_dict(torch.load(output_folder + model_name, map_location="cpu")['model'])
    model.set_actnorm_init()
    model = model.to(device)
    model = model.eval()

    for x,y in test_loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            z, bpd, y_logits = model(x, y_onehot=y)  # return: z, bpd, y_logits
            predict_x = model(y_onehot=y, z=z, temperature=1, reverse=True)
        break
    # print('z=',z,'bpd=', bpd,'y_logits', y_logits)

    print('x=', predict_x)
    print(predict_x.size())
    save_image(x, 'origin_x.png')
    save_image(predict_x, 'predict_x.png')
