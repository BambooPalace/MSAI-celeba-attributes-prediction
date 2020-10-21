# Description
CelebA face attributes prediction using [Large-scale CelebFaces Attributes (CelebA) Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

The model is based on pretrained Resnet18 + 1fc layer.

## Requirements
* Anaconda3
* Pytorch
* Torchvision

## Data
* Training, validation and test data in .jpg format is downloaded from [here](https://drive.google.com/drive/folders/0B7EVK8r0v71peklHb0pGdDl6R28?usp=sharing) into `img_align_celeba` folder for the training, the dataloader in `utils.py` will load the datasets according to the `annotations` files. 
* Private unlabelled dataset is downloaded from [here](https://drive.google.com/file/d/1VF7BkaII4eBZe98v9UNUj_1a2iWGU1kZ/view?usp=drivesdk) into 'test_data/imgae_folder' for labelling.
* Data augmentation(random rotation, flip, >0.8 scale crop) is performed

## Files
```
├── My\ acv\ project.ipynb  #illustration file for code preparation
├── README.md
├── annotations
│   ├── test_attr_list.txt
│   ├── train_attr_list.txt
│   └── val_attr_list.txt
├── celeba.py        #CelebA dataset class
├── checkpoints
│   └── checkpoint_best.pth
├── img_align_celeba        #cropped and aligned jpg dataset
├── log.txt          #accuracy log for 20epoches, the lr displayed is the initial setting
├── loss.py          #focal loss
├── main.py
├── normalize.py     #compute train datasets mean and std
├── test_data
│   └── 13233_imgs        #unlabelled test data
└── utils.py 
```

## Running the codes
* Restart model training (pretrained Resnet18 finetuning) for 20 epoches:
```
python main.py --train_conv --batch_size=512 --epoches=20 --lr=0.1 --checkpoint='checkpoint_best.pth' 
```
* Resume model training for another 20 epoches:
```
python main.py --train_conv --batch_size=512 --epoches=20 --lr=0.1 --checkpoint='checkpoint_best.pth'  --resume
```
* Retrieve test accuracy of lastest trained model:
```
python main.py --batch_size=512 --checkpoint='checkpoint_best.pth'  --test-mode
```
* Make predictions for the unlabelled data into `predictions.txt`:
```
python main.py --batch_size=512 --checkpoint='checkpoint_best.pth'  --test-unlabelled
```

## Results
After running the model with the selected hyperparameters on the total dataset for 20 epochs, the resulting average validation accuracy is 91.8% and average test accuracy is 91.4%. 

The test accuracies for 40 attributes is as below:
       [0.9482, 0.8340, 0.8293, 0.8538, 0.9893, 0.9600, 0.7152, 0.8398, 0.9002, 
        0.9596, 0.9632, 0.8904, 0.9260, 0.9589, 0.9639, 0.9962, 0.9744, 0.9833,
        0.9184, 0.8758, 0.9813, 0.9377, 0.9696, 0.8766, 0.9619, 0.7556, 0.9706,
        0.7684, 0.9381, 0.9504, 0.9775, 0.9283, 0.8448, 0.8488, 0.9031, 0.9904,
        0.9385, 0.8739, 0.9682, 0.8886]

Among test accuracies for 40 attributes, the minimum accuracy is 71.5% for attribute 7 and the maximum accuracy is 99.6% for attribute 16.
Details can refer to [log.txt](https://github.com/BambooPalace/Celeba-attributes-prediction/blob/main/log.txt)

## Readings
-  Z. Liu et al. Deep Learning Face Attributes in the Wild, ICCV 2015
- T-Y Lin et al., Focal Loss for Dense Object Detection, ICCV 2017
- [Face attribute prediction]( https://github.com/d-li14/face-attribute-prediction)
- [Pytorch transfer learning tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Multilabel image classification](https://www.learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/)

