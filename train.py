

#sources :https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#load-data
#https://matplotlib.org/gallery/lines_bars_and_markers/barh.html#sphx-glr-gallery-lines-bars-and-markers-barh-py
#https://docs.python.org/3.2/library/argparse.html
#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import copy
import argparse
import os



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def generate_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, action="store", default="flowers")
    parser.add_argument('--model_path', dest="model_path", action="store", default="my_checkpoint.pth")
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=4096)
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.0003)
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=100)
    return parser.parse_args()


parser = generate_argparse()
image_path = parser.image_path
model_path =parser.model_path
arch = parser.arch
hidden_units = parser.hidden_units
epochs=parser.epochs


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

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)



image_datasets = {x: datasets.ImageFolder(os.path.join(image_path, x),data_transforms[x]) for x in ['train', 'valid','test']}

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,shuffle=True) for x in ['train', 'valid','test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid','test']}
class_names = image_datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def load_model(arch =arch, num_labels=102, hidden_units=4096):

    if  arch=='vgg16':

        model = models.vgg16(pretrained=True)
    elif arch=='alexnet':
        model = models.alexnet(pretrained=True)
    else:
        raise ValueError('Unexpected network architecture', arch)

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



#https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def train_model(model, hidden_units , epochs,learning_rate):


    model.cuda()
    print('the model architecture:', arch)
    print('Number of hidden units:', hidden_units)
    print('Number of epochs:', epochs)
    print('Learning rate:', learning_rate)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() and device == 'gpu':
            model.cuda()


    since =time.time()
    criterion = nn.CrossEntropyLoss()
    optimizer= optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=0.01, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0


    for epoch in range(epochs):

        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)


        for phase in ['train','valid']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            corrects = 0


            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)


                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)


                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


    epoch_loss = running_loss / dataset_sizes[phase]
    epoch_acc = running_corrects.double() / dataset_sizes[phase]

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

    if phase == 'valid' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())

    print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)

    model.class_to_idx = image_datasets['train'].class_to_idx


    return model


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



    loaded_model =load_model(arch =arch , num_labels=102, hidden_units=4096)
    trained_model =train_model(loaded_model, hidden_units , epochs,learning_rate = 0.000)

    loaded_model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {
        'arch': arch,
        'class_to_idx': trained_model.class_to_idx,
        'state_dict': trained_model.state_dict(),
        'hidden_units': 4096
    }

    torch.save(checkpoint, 'my_checkpoint.pth.tar')


if __name__=='__main__':
    main()
