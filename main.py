import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from random import randint
from celeba import CelebA
from utils import *
from loss import FocalLoss

args = parser()

def main():

    print('model settings:\n', args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('training with device:', device)

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
        print('resume from last checkpoint')
        state_dict = torch.load(args.root+'checkpoints/'+args.checkpoint)
        model.load_state_dict(state_dict)
    model.to(device)

    #freeze conv layer parameters if args.train_conv is false, other wise set requires_grad=True
    for params in model.parameters():
        params.requires_grad = args.train_conv

    if args.train_conv:
        parameters = model.parameters()
    else:
        parameters = model.fc.parameters()

    # 3. train and validate the model
    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'focal':
        criterion = FocalLoss(alpha=args.alpha, gamma=args.gamma)#alpha=1 means no emphasis on 0 or 1, smaller gamma means less emphasis on minor probs
    optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    print('model training starts:')

    # 4. test model accuracy on test set
    if args.test_mode:
        test(model, test_loader, criterion, device)
    else:
        model = train_validate(model, criterion, optimizer, scheduler, train_loader, val_loader, device)

        test(model, test_loader, criterion, device)



def train_validate(model, criterion, optimizer, scheduler, train_loader, val_loader, device, validate=True):
    start = time.time()

    best_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    current_lr = optimizer.state_dict()['param_groups'][0]['lr']

    epoches = args.epoches
    for epoch in range(0, epoches):
        print('epoch', epoch + 1, '/', epoches, '\tlearning rate',current_lr )
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
        print('running loss:\t', running_loss, 'training accuracy:\t', train_acc * 100)

        scheduler.step()

        if validate:  # validation if set true
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
                print('validation loss:\t', running_loss, 'validation accuracy:\t', val_acc * 100)

        # copy model weights if test accuracy improves
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, args.root + 'checkpoints/'+args.checkpoint)
        elif val_acc < best_acc - 0.01:  # if validation accuracy drops by 1%, stop the rounds
            print('validation accuracy is not improving at {} epoches, stop early here'.format(epoch))
            break

    print('best accuracy', best_acc * 100)

    # load best weights
    model.load_state_dict(best_weights)
    return model




def test(model, loader, criterion, device):
    state_dict = torch.load(args.root + 'checkpoints/'+args.checkpoint)
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


def output_labels(model, loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        running_loss = test_accuracy = 0
        arr = torch.zeros((1, 40))
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            model.eval()
            s = model(x)
            preds = torch.sigmoid(s).round()  # prob 0.5 is 1
            preds = preds * 2 - 1
            arr = np.concatenate((arr, preds.cpu()), axis=0)
    return arr[1:]



def test_unlabelled():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    state_dict = torch.load(args.root+'checkpoints/'+args.checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)

    print('Predicting labels for the private test data')
    labels = output_labels(model, final_loader)
    # save in case code breaks
    torch.save(labels,'labels.pt')
    np.savetxt("predictions.txt", labels, fmt='%d', footer='\n', comments='')

    img_names = os.listdir(args.root+'test_data/13233_imgs/')
    file = "predictions.txt"

    with open(file, 'r') as f:
        lines = [' '.join([img, x.strip()+'\n']) for x, img in zip(f.readlines(), img_names)]

    with open(file, 'w') as f:
        f.writelines(lines)

    print('labels has been saved in the "predictions.txt" in the current directory')




if __name__ == '__main__':
    if args.test_unlabelled:
        test_unlabelled()
    else:
        main()

