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

def compute_nll(dataset, model, dataloader):
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=6)

    nlls = []
    iter = 0
    for x,y in dataloader:
        iter += 1
        x = x.to(device)

        if hparams['y_condition']:
            y = y.to(device)
        else:
            y = None

        with torch.no_grad():
            _, nll, _ = model(x, y_onehot=y)  # return: z, bpd, y_logits
            nlls.append(nll)
        print(iter)

    return torch.cat(nlls).cpu()


if __name__ == "__main__":
    device = torch.device("cuda")

    output_folder = '0614_0641_logs/'
    model_name = 'glow_checkpoint_75000.pt'

    with open(output_folder + 'hparams.json') as json_file:
        hparams = json.load(json_file)


    # image_shape, num_classes, _, test_cifar = get_CIFAR10(hparams['augment'], hparams['dataroot'], True)
    # image_shape, num_classes, _, test_svhn = get_SVHN(hparams['augment'], hparams['dataroot'], True)
    image_shape = (64,64,3)
    num_classes = 40
    dataset_test = CelebALoader(root_folder=hparams['dataroot']) #'/home/yellow/deep-learning-and-practice/hw7/dataset/task_2/'
    test_loader = DataLoader(dataset_test,batch_size=hparams['batch_size'],shuffle=True,drop_last=True)
    model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
                 hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes,
                 hparams['learn_top'], hparams['y_condition'])


    model.load_state_dict(torch.load(output_folder + model_name, map_location="cpu")['model'])
    model.set_actnorm_init()

    model = model.to(device)

    model = model.eval()


    nll = compute_nll(dataset_test, model, test_loader)
    print("NLL", torch.mean(nll))


    plt.figure(figsize=(20,10))
    plt.title("Histogram Glow ")
    plt.xlabel("Negative bits per dimension")
    plt.hist(-nll.numpy(), label="SMILE", density=True, bins=50)
    plt.legend()
    plt.show()
