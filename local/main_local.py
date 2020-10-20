import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

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


resnet_choices = ['resnet18','resnet34','resnet50']
parser = argparse.ArgumentParser(description='CelebA attributes training')
parser.add_argument('--root', default='/Users/clairegong/Downloads/ACV Project/', type=str,\
                    help='path to the scripts and data folder')
parser.add_argument('--model', default='resnet18',choices=resnet_choices, type=str, help='choice of resnet models 18/34/50')
parser.add_argument('--pretrained', default=True, type=bool, help='set the model to be pretrained or not')
parser.add_argument('--resume', default=False, type=bool, help='from first epoch or resume training of extra no of epoches')
parser.add_argument('--lr', default = 0.01, type=float, help='set learning rate to be 0.01 for default to train the conv layers,\
    if freeze conv layer suggest use smaller values')
parser.add_argument('--train_conv', default=True, type=bool, help='to retrain the conv layers of convnet or to freeze it')
parser.add_argument('--epoches', default=10, type=int, help='number of epoches')
parser.add_argument('--batch_size', default=64, type=int, help='same batch size for train, val and test')
parser.add_argument('--test_mode', action='store_true', help='only check accuracy on test data, no training performed')
parser.add_argument('--test_unlabelled', action='store_true', help='no training, only inference if option activated')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
    print('model settings:\n', args)


    # 1. load the datasets
    train_loader, val_loader, test_loader = dataloaders()
    #show number of positive attributes
    print(positive_attributes(train_loader.dataset))
    print(positive_attributes(val_loader.dataset))
    print(positive_attributes(test_loader.dataset))

    # 2. retrieve the pretrained model
    model = load_model()
    #if resume is true, load the previously save checkpoint
    if args.resume:
        state_dict = torch.load(args.root+'checkpoints/checkpoint.pth')
        model.load_state_dict(state_dict)
    model.to(device)

    if args.train_conv:
        parameters = model.parameters()
    else:
        parameters = model.fc.parameters()

    # 3. train and validate the model
    # criterion = nn.BCEWithLogitsLoss()
    from loss import FocalLoss
    criterion = FocalLoss(alpha=1, gamma=1)

    optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 4. test model accuracy on test set

    if args.test_mode:
        test(model, test_loader, criterion, device)
    else:
        model = train_validate(model, criterion, optimizer, scheduler, train_loader, val_loader, device)

        test(model, test_loader, criterion, device)



def dataloaders():
    # Data loading code
    # 1. all mages are already aligned to 218*178;

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = CelebA(
        args.root,
        'train600.txt',
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
        'val200.txt',
        transforms.Compose([
            transforms.Resize((178, 178)),
            #         transforms.CenterCrop((178,178)),
            transforms.ToTensor(),
            normalize,
        ]))

    test_dataset = CelebA(
        args.root,
        'test200.txt',
        transforms.Compose([
            transforms.Resize((178, 178)),
            #         transforms.CenterCrop((178,178)),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
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

    for params in model.parameters():
        params.requires_grad = args.train_conv

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

def train_validate(model, criterion, optimizer, scheduler, train_loader, val_loader, validate=True):
        start = time.time()

        best_weights = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        epoches = args.epoches
        for epoch in range(0, epoches):
            print('epoch', epoch + 1, '/', epoches)
            running_loss = train_acc = 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                model.train()
                optimizer.zero_grad()

                x.requires_grad_()
                s = model(x)
                loss = criterion(s, y.float())
                loss.backward()
                optimizer.step()

                running_loss += loss.detach().item() * x.size(0)
                train_acc += batch_accuracy(s.detach(), y)[0]
            running_loss /= len(train_loader.dataset)
            train_acc /= len(train_loader.dataset)
            print('time elapsed:%d s ' % (time.time() - start))
            print('running loss:', running_loss, 'training accuracy:', train_acc * 100)

            scheduler.step()

            if validate:#validation if set true
                with torch.no_grad():
                    running_loss = val_acc = 0
                    for x, y in val_loader:
                        x, y = x.to(device), y.to(device)
                        model.eval()
                        s = model(x)
                        loss = criterion(s, y.float())

                        running_loss += loss.detach().item() * x.size(0)
                        val_acc += batch_accuracy(s.detach(), y)[0]
                    running_loss /= len(val_loader.dataset)
                    val_acc /= len(val_loader.dataset)
                    print('validation loss:', running_loss, 'validation accuracy:', val_acc * 100)

            # copy model weights if test accuracy improves
            if val_acc > best_acc:
                best_acc = val_acc
                best_weights = copy.deepcopy(model.state_dict())
            elif epoch>9 :#if validation accuracy does not improve after 10 rounds, stop the rounds
                break

        print('best accuracy', best_acc * 100)

        # load best weights and save to checkpoint
        model.load_state_dict(best_weights)
        torch.save(best_weights, args.root+'checkpoints/checkpoint.pth')
        return model

def test(model, loader, criterion, device):
    state_dict = torch.load(args.root + 'checkpoints/checkpoint.pth')
    model.load_state_dict(state_dict)

    start = time.time()
    with torch.no_grad():
        running_loss = test_accuracy = 0
        attr_accuracy = torch.zeros(40).to(device)
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            model.eval()
            s = model(x)
            loss = criterion(s, y.float())

            running_loss += loss.detach().item() * x.size(0)
            test_accuracy += batch_accuracy(s.detach(), y)[0]
            #attr_acc need to be on cuda for computation
            attr_accuracy += batch_accuracy(s.detach(), y)[1]
        running_loss /= len(loader.dataset)
        test_accuracy /= len(loader.dataset)
        attr_accuracy /= len(loader.dataset)
        print('time elapsed:%d s ' % (time.time() - start))
        print('test loss:\t', running_loss, 'test accuracy:\t', test_accuracy * 100)
        print('accuracy per attribute\n', attr_accuracy)
        print(rank(attr_accuracy))

        return test_accuracy, attr_accuracy

def positive_attributes(dataset):
    #dim tip, arr dim is (200,40), put 0/row then calculation cross 200 rows, result size will be same as dim 1-> 40
    distribution = np.array(dataset.targets).sum(0)
    sort_idx = np.argsort(distribution)+1
    print('for {} samples, min num {} for attribute {} and max num {} for attribute {}'.format(len(dataset),distribution.min(), sort_idx[0],\
                                                                               distribution.max(), sort_idx[-1]))
    return sort_idx

def rank(acc):
    sort_idx = np.argsort(acc)+1
    print('Among test accuracies for 40 attributes, min accuray is {:.2f} for attribute {} and max accuracy is {:.2f} for attribute {}'.format(acc.min(), sort_idx[0],\
                                                                               acc.max(), sort_idx[-1]))
    return sort_idx

def output_labels(model, loader):
    #     state_dict = torch.load('checkpoint0.pth')
    #     model.load_state_dict(state_dict)

    with torch.no_grad():
        running_loss = test_accuracy = 0
        arr = torch.zeros((1, 40))
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            model.eval()
            s = model(x)
            preds = torch.sigmoid(s).round()  # prob 0.5 is 1
            preds = preds * 2 - 1
            arr = np.concatenate((arr, preds), axis=0)
    return arr[1:]



def test_unlabelled():

    #load data
    final_data = datasets.ImageFolder(args.root + 'test_data/', transform=transforms.Compose([
        transforms.Resize((178, 178)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]))
    final_loader = torch.utils.data.DataLoader(final_data, shuffle=False, batch_size=32)

    #load model architecture and load parameters from checkpoint
    model = load_model()
    state_dict = torch.load(args.root+'checkpoints/checkpoint.pth')
    model.load_state_dict(state_dict)
    model.to(device)

    print('Predicting labels for the private test data')
    labels = output_labels(model, final_loader)
    #save in case code breaks
    torch.save(labels, 'labels.pt')
    np.savetxt("predictions.txt", labels, fmt='%d', footer='\n', comments='')

    img_names = os.listdir(args.root+'test_data/13234_imgs/')
    file = "predictions.txt"

    with open(file, 'r') as f:
        lines = [' '.join([img, x.strip(), '\n']) for x, img in zip(f.readlines(), img_names)]

    with open(file, 'w') as f:
        f.writelines(lines)

    print('labels has been saved in the "predictions.txt" in the current directory')




if __name__ == '__main__':
    if args.test_unlabelled:
        test_unlabelled()
    else:
        main()