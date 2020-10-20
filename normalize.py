from celeba import CelebA
import torch
from torchvision import datasets, models, transforms


def dataloaders():
    # Data loading code
    # 1. all mages are already aligned to 218*178;

    normalize0 = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = CelebA(
        '/home/MSAI/cgong002/acv_project_celeba/',
        'train_attr_list.txt',
        transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            #         transforms.Resize((178,178)),
            #         transforms.CenterCrop((178,178)),
            transforms.RandomResizedCrop(178, scale=(0.8, 1.0)),
            # should not cut many info for multi label classification
            transforms.ToTensor(),
            # normalize,
        ]))



    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True,num_workers=7, pin_memory=True)
    return train_loader
train_loader = dataloaders()
loader = train_loader

mean = 0.
std = 0.
for images, _ in loader:
    batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)

mean /= len(loader.dataset)
std /= len(loader.dataset)
print(mean)
print(std)