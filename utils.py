import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from random import randint
from celeba import CelebA

def parser():
    resnet_choices = ['resnet18','resnet34','resnet50']
    parser = argparse.ArgumentParser(description='CelebA attributes training')
    parser.add_argument('--root', default='/home/MSAI/cgong002/acv_project_celeba/', type=str,\
                        help='path to the scripts and data folder')
    parser.add_argument('--checkpoint', default='checkpoint.pth',type=str,help='checkpoint file name to save and load checkpoint')
    parser.add_argument('--loss', default='bce', choices=['bce', 'focal'],type=str, help='choose loss function')
    parser.add_argument('--model', default='resnet18',choices=resnet_choices, type=str, help='choice of resnet models 18/34/50')
    parser.add_argument('--pretrained', default=True, type=bool, help='set the model to be pretrained or not')
    parser.add_argument('--resume', action='store_true', help='from first epoch or resume training of extra no of epoches')
    parser.add_argument('--lr', default = 0.01, type=float, help='set learning rate to be 0.01 for default to train the conv layers,\
        if freeze conv layer suggest use smaller values i.e. 0.001')
    parser.add_argument('--train_conv', action='store_true', help='to retrain the conv layers of convnet or to freeze it')
    parser.add_argument('--epoches', default=10, type=int, help='number of epoches')
    parser.add_argument('--batch_size', default=256, type=int, help='same batch size for train, val and test')
    parser.add_argument('--test_mode', action='store_true', help='only check accuracy on test data, no training performed')
    parser.add_argument('--test_unlabelled', action='store_true', help='no training, only inference if option activated')
    parser.add_argument('--alpha', default=0.25, type=float, help='alpha of loss function')
    parser.add_argument('--gamma',default=2, type=int, help='gamma of focal loss function')
    args = parser.parse_args()
    return args

args = parser()

def dataloaders():
    # Data loading code
    # 1. all mages are already aligned to 218*178;

    normalize_old = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    #newly computed after data augmentation
    normalize_new = transforms.Normalize([0.4807, 0.3973, 0.3534],
                                         [0.2838, 0.2535, 0.2443])
    normalize = normalize_new

    train_dataset = CelebA(
        args.root,
        'train_attr_list.txt',
        transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            #         transforms.Resize((178,178)),
            #         transforms.CenterCrop((178,178)),
            transforms.RandomResizedCrop(178, scale=(0.8, 1.0)),
            # should not cut many info for multi label classification
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = CelebA(
        args.root,
        'val_attr_list.txt',
        transforms.Compose([
            transforms.Resize((178, 178)),
            #         transforms.CenterCrop((178,178)),
            transforms.ToTensor(),
            normalize,
        ]))

    test_dataset = CelebA(
        args.root,
        'test_attr_list.txt',
        transforms.Compose([
            transforms.Resize((178, 178)),
            #         transforms.CenterCrop((178,178)),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=7, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=7, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=7, pin_memory=True)
    return train_loader, val_loader, test_loader

def load_model():
    if args.model == 'resnet18':
        model = models.resnet18(pretrained=args.pretrained)
    elif args.model == 'resnet34':
        model = models.resnet34(pretrained=args.pretrained)
    elif args.model == 'resnet50':
        model = models.resnet50(pretrained=args.pretrained)
    else:
        print('input model is not an existing resnet architecture')

    fc_size = model.fc.in_features
    #reset fc layer size for celeba attribute dataset: 40 labels
    model.fc = nn.Linear(fc_size, 40)
    #save model starting state in case of retraining the model
    checkpoint0 = torch.save(model.state_dict(), args.root+'checkpoints/checkpoint0.pth')

    return model

def batch_accuracy(scores, labels):
    preds = torch.sigmoid(scores).round()
    truth = (preds==labels)
    num_correct = truth.float().sum().item()/40 #avg no of  correct prediction * no of samples
    num_correct_attr = truth.float().sum(0)#40-size tensor
    return num_correct, num_correct_attr

def positive_attributes(dataset):
    #dim tip, arr dim is (200,40), put 0/row then calculation cross 200 rows, result size will be same as dim 1-> 40
    distribution = np.array(dataset.targets).sum(0)
    sort_idx = np.argsort(distribution)+1
    print('for {} samples, min num {} for attribute {} and max num {} for attribute {}'.format(len(dataset),distribution.min(), sort_idx[0],\
                                                                               distribution.max(), sort_idx[-1]))
    return sort_idx

def rank(acc):
    sort_idx = np.argsort(acc.cpu())+1
    print('Among test accuracies for 40 attributes, min accuray is {:.2f} for attribute {} and max accuracy is {:.2f} for attribute {}'.format(acc.min(), sort_idx[0],\
                                                                               acc.max(), sort_idx[-1]))
    print('Ranked attributes from small to large accuracy.')
    return sort_idx