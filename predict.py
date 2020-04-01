# Imports here
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict
from PIL import Image
import numpy as np
import seaborn as sns
import json


def label_mapping():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    return cat_to_name


def load_checkpoint(filepath):
    # load saved model on GPU when cuda is available, otherwise on CPU
    # https://discuss.pytorch.org/t/problem-loading-model-trained-on-gpu/17745
    checkpoint = torch.load(filepath) if torch.cuda.is_available() \
                            else torch.load(filepath, map_location=lambda storage, loc: storage)
    
    model = models.vgg16(pretrained=True)
        
    model.class_to_idx = checkpoint['class_map']
    model.classifier = checkpoint['classifier']
    
    model.load_state_dict(checkpoint['state_dict'])
    
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']
    
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, optimizer, epochs


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    # load image
    img = Image.open(image)
    
    # resize image 256 pixels
    w, h = img.size
    img = img.resize((256, int(256 * h/w))) if w < 256 else img.resize((int(256 * w/h), 256))
    
    # crop image 224 x 224 by center
    # https://stackoverflow.com/questions/16646183/crop-an-image-in-the-centre-using-pil  
    left = (img.size[0] - 224)/2
    bottom = (img.size[1] - 224)/2
    right = (img.size[0] + 224)/2
    top = (img.size[1] + 224)/2

    # Crop the center of the image
    crop_img = img.crop((left, bottom, right, top))
    
    # np_image
    np_image = np.array(crop_img) / 255.0

    # Normalization
    means = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    norm_image = (np_image - means) / std
    
    # reorder
    processed_image = norm_image.transpose((2, 0, 1))
    
    return processed_image


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # https://pytorch.org/docs/stable/torch.html
    inputs = torch.from_numpy(process_image(image_path)).type(torch.FloatTensor).unsqueeze(0)
    
    inputs = inputs.to(device)
    logps = model.forward(inputs)

    ps = torch.exp(logps)
    top_prob, top_class = ps.topk(topk, dim=1)
    
    # https://stackoverflow.com/questions/49768306/pytorch-tensor-to-numpy-array
    top_probs = top_prob.cpu().detach().numpy().tolist()[0]
    top_class = top_class.cpu().detach().numpy().tolist()[0]

    # convert index to class
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    # get class labels from top_class value
    classes = [idx_to_class[val] for val in top_class]
    
    return top_probs, classes


# TODO: Display an image along with the top 5 classes
def plot_predicts(image_path, model, topk=5):

    cat_to_name = label_mapping()
    img =  process_image(image_path)
    probs, classes = predict(image_path, model, topk)
    names = [cat_to_name[i] for i in classes]

    # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.figure.html
    plt.figure(figsize=(4, 10))
    # https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.subplot.html
    ax = plt.subplot(211)

    plt.axis('off') # https://stackoverflow.com/questions/9295026/matplotlib-plots-removing-axis-legends-and-white-spaces
    title = ax.set_title(names[0])
    imshow(img, ax, title)

    plt.subplot(212)
    # https://seaborn.pydata.org/generated/seaborn.barplot.html
    # https://seaborn.pydata.org/tutorial/color_palettes.html
    sns.barplot(x=probs, y=names, color=sns.color_palette("Blues")[-2]) 

    # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots_adjust.html
    plt.subplots_adjust(left=0.125, right=0.9, bottom=0.3, top=1.0, hspace = 0.05)
    plt.show()

    
def main():
    model, _, _ = load_checkpoint('checkpoint.pth')
    image_path = 'flower' + '/test/11/' + 'image_03141.jpg'
    plot_predicts(image_path, model, topk=5)
    

if __name__ == '__main__':
    main()
    