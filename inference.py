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
from torchvision import transforms
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage, Loss

from mydataset import get_CelebA_data, CelebALoader, get_test_conditions, get_new_test_conditions, CLEVRDataset
from model import Glow
from utils import save_image
from evaluator import evaluation_model


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
    save_image(x, 'images/task2_origin.png',normalize=True)
    save_image(predict_x, 'images/task2_inverse.png',normalize=True)





    output_folder = '0616_0611_logs_task1/'
    model_name = 'glow_checkpoint_115210.pt'

    with open(output_folder + 'hparams.json') as json_file:
        hparams = json.load(json_file)

    image_shape = (64,64,3)
    num_classes = 24
    dataset_test = CLEVRDataset(root_folder=hparams['dataroot'],img_folder=hparams['dataroot']+'images/')
    test_loader = DataLoader(dataset_test,batch_size=32,shuffle=True,drop_last=True)
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
    save_image(x, 'images/task1_origin.png', normalize=True)
    save_image(predict_x, 'images/task1_inverse.png', normalize=True)


    def norm_ip(img, min, max):
        img.clamp_(min=min, max=max)
        img.add_(-min).div_(max - min + 1e-5)

    def norm_range(t, range):
        if range is not None:
            norm_ip(t, range[0], range[1])
        else:
            norm_ip(t, float(t.min()), float(t.max()))


    evaluator = evaluation_model("/home/yellow/deep-learning-and-practice/hw7/classifier_weight.pth")
    test_conditions = get_test_conditions(hparams['dataroot']).cuda()
    tmp_x = torch.rand( ( len(test_conditions) , image_shape[2], image_shape[0], image_shape[0]) ).cuda()
    z, _, _ = model(tmp_x, test_conditions)
    z = torch.randn(z.size()).cuda()
    predict_x = model(y_onehot=test_conditions, z=z, temperature=1, reverse=True)
    for t in predict_x:  # loop over mini-batch dimension
        norm_range(t, None)
    score = evaluator.eval(predict_x, test_conditions)
    save_image(predict_x, f"score{score:.3f}.png")

    new_test_conditions = get_new_test_conditions(hparams['dataroot']).cuda()
    new_predict_x = model(y_onehot=new_test_conditions, z=z, temperature=1, reverse=True)
    for t in predict_x:  # loop over mini-batch dimension
        norm_range(t, None)
    new_score = evaluator.eval(new_predict_x, new_test_conditions)
    save_image(predict_x, f"newscore{new_score:.3f}.png")
