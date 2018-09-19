
# coding: utf-8

# # Developing an AI application
#
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.
#
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below.
#
# <img src='assets/Flowers.png' width=500px>
#
# The project is broken down into multiple steps:
#
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
#
# We'll lead you through each part which you'll implement in Python.
#
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
#
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

# In[1]:


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
##https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#load-data
##https://matplotlib.org/gallery/lines_bars_and_markers/barh.html#sphx-glr-gallery-lines-bars-and-markers-barh-py


# ## Load the data
#
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
#
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
#
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#

# In[2]:


data_dir = 'flowers'


# In[3]:




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


# ### Label mapping
#
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[4]:


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x]) for x in ['train', 'valid','test']}

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,shuffle=True) for x in ['train', 'valid','test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid','test']}
class_names = image_datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('Train_Size:',dataset_sizes['train'])
print('Valid_Size:',dataset_sizes['valid'])
print('Test_Size:',dataset_sizes['test'])


# # Building and training the classifier
#
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
#
# We're going to leave this part up to you. If you want to talk through it with someone, chat with your fellow students! You can also ask questions on the forums or join the instructors in office hours.
#
# Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
#
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
#
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
#
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.

# In[5]:



arch = {'vgg16': 25088, 'alexnet' : 9216}


def load_model(arch='vgg16', num_labels=102, hidden_units=4096):

   if arch == 'vgg16':
       model = models.vgg16(pretrained=True)
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







#https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def train_model(model, criterion, optimizer, scheduler, num_epochs=15, device = 'gpu',learning_rate=0.001  ):


   since =time.time()
   criterion = nn.CrossEntropyLoss()
   optimizer= optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=0.01, momentum=0.9)
   scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

   model.cuda()



   best_model_wts = copy.deepcopy(model.state_dict())

   best_acc = 0.0


   for epoch in range(num_epochs):

       print('Epoch {}/{}'.format(epoch, num_epochs - 1))
       print('-' * 10)


       for phase in ['train','valid']:
           if phase == 'train':
               scheduler.step()
               model.train()
           else:
               model.eval()
           running_loss = 0.0
           running_corrects = 0


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

           print('{} Loss: {:.4f} Acc: {:.4f}'.format(
               phase, epoch_loss, epoch_acc))


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
model=load_model(arch= 'vgg16', num_labels=102, hidden_units=4096)
criterion = nn.CrossEntropyLoss()
optimizer= optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=0.01, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
fit_model = train_model(model, criterion, optimizer, scheduler, num_epochs=15, device = 'gpu',learning_rate=0.001  )
best_model_wts = copy.deepcopy(model.state_dict())





# ## Testing your network
#
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# In[6]:


def check_accuracy_on_test(testloader):
    correct = 0
    total = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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


check_accuracy_on_test(dataloaders['test'])


# ## Save the checkpoint
#
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
#
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
#
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[7]:


model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {
    'arch': 'vgg16',
    'class_to_idx': model.class_to_idx,
    'state_dict': model.state_dict(),
    'hidden_units': 4096
}

torch.save(checkpoint, 'vgg16chekpoint.pth.tar')


# ## Loading the checkpoint
#
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# In[8]:


def load_checkpoint(model_path):

    torch.load(model_path)
    arch = checkpoint['arch']
    num_labels = len(checkpoint['class_to_idx'])
    hidden_units = checkpoint['hidden_units']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

loaded_model = load_checkpoint('vgg16chekpoint.pth.tar')

print(loaded_model)






# # Inference for classification
#
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like
#
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
#
# First you'll need to handle processing the input image such that it can be used in your network.
#
# ## Image Preprocessing
#
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training.
#
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
#
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
#
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation.
#
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

# In[9]:


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
image = ('flowers/test/101/image_07983.jpg')
image = process_image(image)
print(image.shape)



# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# In[10]:



#https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

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

# ### Using the image datasets, define the dataloaders
dataloaders = {
    x: data.DataLoader(image_datasets[x], batch_size=1, shuffle=True, num_workers=2)
    for x in list(image_datasets.keys())
}

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

labels = list(cat_to_name.values())

imshow(out, title=[labels[x] for x in classes])



# ## Class Prediction
#
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
#
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
#
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
#
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[79]:


model.class_to_idx = image_datasets['train'].class_to_idx

def predict(image_path, model, topk=5, device='gpu'):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.cuda()
    tensor_image = process_image(image_path)
    tensor_image = tensor_image.unsqueeze_(0)
    tensor_image= tensor_image.float()
    if device == 'gpu':
        model.to('cuda:0')
        tensor_image = Variable(tensor_image.cuda())
    else:
        tensor_image = Variable(tensor_tensor)

    with torch.no_grad():
        output = model(tensor_image)

    prblty = F.softmax(output.data,dim=1)

    probs, classes= prblty.topk(topk)

    labels = list(cat_to_name.values())

    classes = [labels[int(x)] for x in classes[0]]

    return  probs,classes


probs, classes =predict('flowers/train/10/image_07087.jpg', loaded_model, topk=5, device='gpu')


print(classes)


print(probs)


# ## Sanity Checking
#
# Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:
#
# <img src='assets/inference_example.png' width=300px>
#
# You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.

# In[82]:


###https://matplotlib.org/gallery/lines_bars_and_markers/barh.html#sphx-glr-gallery-lines-bars-and-markers-barh-py

img = mpimg.imread(image_path)
plt.rcdefaults()
f, plot = plt.subplots(2,1)
probs, classes = predict('flowers/train/10/image_07087.jpg', loaded_model, topk=5, device='gpu')

probs= probs.cpu()
probs = probs.data.numpy()

print(probs[0])
print(classes)

plot[0].imshow(img)
plot[0].set_title(classes[0])

y_pos = np.arange(len(classes))
plot[1].barh(y_pos, probs[0], align='center', color='green')
plot[1].set_yticks(y_pos)
plot[1].set_yticklabels(classes)
plot[1].invert_yaxis()
_ = plot[1].set_xlabel('Probability')
