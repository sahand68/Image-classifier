
# sources :https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#load-data\
#https://docs.python.org/3.2/library/argparse.html

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
import os
import time
import copy

from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

data_dir = 'flowers'
data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         ])
    }

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x]) for x in ['train', 'valid','test']}

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,shuffle=True) for x in ['train', 'valid','test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid','test']}
class_names = image_datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im= Image.open(image)
    process = transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean =[0.485, 0.456, 0.406],
                                                       std=[0.229, 0.224, 0.225])])
    tensor_image = process(im)

    return tensor_image

def load_model(arch='vgg16', num_labels=102, hidden_units=4096):

   if arch == 'vgg16':
       model = models.vgg16(pretrained= True)
   elif arch== 'alexnet':
       model = models.alexnet(pretrained = True)
   else:

       print('please choose vgg16 or alexnet')

   for param in model.parameters():
       param.requires_grad = False

   features = list(model.classifier.children())[:-1]

   num_filters = model.classifier[len(features)].in_features


   features.extend([
       nn.Dropout(),
       nn.Linear(num_filters, hidden_units),
       nn.ReLU(True),
       nn.Dropout(),
       nn.Linear(hidden_units, hidden_units),
       nn.ReLU(True),
       nn.Linear(hidden_units, num_labels)],)


   model.classifier = nn.Sequential(*features)
   return model


def load_checkpoint(model_path):
    model =load_model(arch = 'vgg16', num_labels=102, hidden_units=4096)
    loaded_model =torch.load(model_path)

    arch = loaded_model['arch']
    num_labels = len(loaded_model['class_to_idx'])
    hidden_units = loaded_model['hidden_units']
    model.load_state_dict(loaded_model['state_dict'])
    model.class_to_idx = loaded_model['class_to_idx']
    model.eval()
    return model

def imshow(inp, ax =None,  title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    ax.imshow(inp)
    if title is not None:
        plt.title(title)

    return ax



def predict(image_path, model_path,topk=5, device=device):

    tensor_image = process_image(image_path)
    tensor_image = tensor_image.unsqueeze_(0)
    tensor_image= tensor_image.float()
    model =load_checkpoint(model_path)
    if device == 'gpu':
        model.to('cuda:0')
        tensor_image = Variable(tensor_image.cuda())
    else:
        tensor_image = Variable(tensor_image)

    with torch.no_grad():
        output = model(tensor_image)

    prblty = F.softmax(output.data,dim=1)

    probs, classes= prblty.topk(topk)

    labels = list(cat_to_name.values())

    classes = [labels[int(x)] for x in classes[0]]
    probs= probs.cpu()
    probs = probs.data.numpy()

    return  probs[0],classes


def generate_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, action="store", default="flowers\\valid\\7\\image_07216.jpg")
    parser.add_argument('--device', dest='device', action="store", default="gpu")
    parser.add_argument('--model_path', dest="model_path", action="store", default="my_checkpoint.pth.tar")
    parser.add_argument('--topk', type=int, default=5, dest = 'topk',help='number of top likey classes to plot')


    return parser.parse_args()


def check_accuracy_on_test(testloader, model_path):
    correct = 0
    total = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model =load_checkpoint(model_path)
    model.cuda()


    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images= images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))




def main():



    data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(45),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
             ]),
            'valid': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
             ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
             ])
        }
    #https://www.programcreek.com/python/example/748/argparse.ArgumentParser
    args = generate_argparse()
    image_path = args.image_path
    model_path =args.model_path
    device =args.device

    model =torch.load(model_path)
    probs, classes = predict(image_path, model_path,topk=5, device='gpu')
    print(probs, classes)
    img = mpimg.imread(image_path)
    plt.rcdefaults()
    f, plot = plt.subplots(2,1)
    plot[0].imshow(img)
    plot[0].set_title(classes[0])

    y_pos = np.arange(len(classes))
    plot[1].barh(y_pos, probs, align='center', color='green')
    plot[1].set_yticks(y_pos)
    plot[1].set_yticklabels(classes)
    plot[1].invert_yaxis()
    _ = plot[1].set_xlabel('Probability')

    check_accuracy_on_test(dataloaders['test'], model_path)

    plt.show()

if __name__=='__main__':
    main()
