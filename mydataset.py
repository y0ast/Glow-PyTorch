import json
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import os
import numpy as np


def get_test_conditions(root_folder):
    """
    :return: (#test conditions,#classes) tensors
    """
    with open(os.path.join(root_folder, 'objects.json'), 'r') as file:
        classes = json.load(file)
    with open(os.path.join(root_folder, 'test.json'), 'r') as file:
        test_conditions_list=json.load(file)

    labels=torch.zeros(len(test_conditions_list),len(classes))
    for i in range(len(test_conditions_list)):
        for condition in test_conditions_list[i]:
            labels[i,int(classes[condition])]=1.

    return labels

def get_new_test_conditions(root_folder):
    """
    :return: (#test conditions,#classes) tensors
    """
    with open(os.path.join(root_folder, 'objects.json'), 'r') as file:
        classes = json.load(file)
    with open(os.path.join(root_folder, 'new_test.json'), 'r') as file:
        test_conditions_list=json.load(file)

    labels=torch.zeros(len(test_conditions_list),len(classes))
    for i in range(len(test_conditions_list)):
        for condition in test_conditions_list[i]:
            labels[i,int(classes[condition])]=1.

    return labels

class CLEVRDataset(data.Dataset):
    def __init__(self, root_folder, img_folder):
        """
        :param img_path: file of training images
        :param json_path: train.json
        """
        self.img_path = img_folder
        self.max_objects=0
        with open(os.path.join(root_folder, 'objects.json'), 'r') as file:
            self.classes = json.load(file)
        self.numclasses=len(self.classes)
        self.img_names=[]
        self.img_conditions=[]
        with open(os.path.join(root_folder, 'train.json'), 'r') as file:
            dict=json.load(file)
            for img_name,img_condition in dict.items():
                self.img_names.append(img_name)
                self.max_objects=max(self.max_objects,len(img_condition))
                self.img_conditions.append([self.classes[condition] for condition in img_condition])
        self.transformations=transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img=Image.open(os.path.join(self.img_path,self.img_names[index])).convert('RGB')
        img=self.transformations(img)
        condition=self.int2onehot(self.img_conditions[index])
        return img,condition

    def int2onehot(self,int_list):
        onehot=torch.zeros(self.numclasses)
        for i in int_list:
            onehot[i]=1.
        return onehot

def get_CelebA_data(root_folder):
    img_list = os.listdir(os.path.join(root_folder, 'CelebA-HQ-img'))
    label_list = []
    f = open(os.path.join(root_folder, 'CelebA-HQ-attribute-anno.txt'), 'r')
    num_imgs = int(f.readline()[:-1])
    attrs = f.readline()[:-1].split(' ')
    for idx in range(num_imgs):
        line = f.readline()[:-1].split(' ')
        label = line[2:]
        label = list(map(int, label))
        for i in range(len(label)):
            if label[i] == -1: label[i]=0
        label_list.append(label)
    f.close()
    return img_list, label_list


class CelebALoader(data.Dataset):
    def __init__(self, root_folder, trans=None, cond=False):
        self.root_folder = root_folder
        assert os.path.isdir(self.root_folder), '{} is not a valid directory'.format(self.root_folder)

        self.cond = cond
        self.img_list, self.label_list = get_CelebA_data(self.root_folder)
        self.num_classes = 40
        self.transformations=transforms.Compose([transforms.Resize((64,64)),\
                                                transforms.ToTensor(),\
                                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        print("> Found %d images..." % (len(self.img_list)))

    def __len__(self):
        return 30000

    def __getitem__(self, index):
        img=Image.open(self.root_folder + 'CelebA-HQ-img/' + self.img_list[index]).convert('RGB')
        img=self.transformations(img)
        label = torch.LongTensor(self.label_list[index])
        return img,label
