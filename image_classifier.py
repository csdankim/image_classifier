import torch
from torch import nn
from torchvision import datasets, transforms, models
from collections import OrderedDict

import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
import seaborn as sns
import json


def load_data(data_path):
    data_dir = data_path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    return train_data, valid_data, test_data, trainloader, validloader, testloader


def build_classifier(model, input_units, hidden_units, dropout):
    # Freeze only features parameters so we don't backprop through them
    # instead of making new classifier, I would like to use the same structure as VGG16,
    # but I just want to train classifier only.
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_units, hidden_units, bias=True)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(dropout)),
        ('fc2', nn.Linear(hidden_units, 128, bias=True)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    return model


def validation(model, device, criterion, validloader):
    valid_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            valid_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    return valid_loss / len(validloader), accuracy / len(validloader)


def save_model(model, train_data, optimizer, save_dir, epochs):
    # TODO: Save the checkpoint
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'class_map': model.class_to_idx,
                  'optimizer': optimizer.state_dict,
                  'classifier': model.classifier,
                  'epochs': epochs,
                  'state_dict': model.state_dict()}

    return torch.save(checkpoint, 'checkpoint.pth', save_dir)


def train(model, device, criterion, optimizer, trainloader, validloader, epochs, print_every):
    steps = 0
    running_loss = 0
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss, valid_accuracy = validation(model, device, criterion, validloader)

                train_loss = running_loss / print_every

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {train_loss:.3f}.. "
                      f"Valid loss: {valid_loss:.3f}.. "
                      f"Valid accuracy: {valid_accuracy:.3f}")
                running_loss = 0
                model.train()

    return train_loss, valid_loss, valid_accuracy


def label_mapping(cat_names_path):
    with open(cat_names_path, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name


def load_checkpoint(model, filepath, gpu_mode):
    # load saved model on GPU when cuda is available, otherwise on CPU
    # https://discuss.pytorch.org/t/problem-loading-model-trained-on-gpu/17745
    checkpoint = torch.load(filepath) if gpu_mode \
        else torch.load(filepath, map_location=lambda storage, loc: storage)

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
    print(img.size)
    # resize image 256 pixels
    w, h = img.size
    img = img.resize((256, int(256 * h / w))) if w < 256 else img.resize((int(256 * w / h), 256))

    # crop image 224 x 224 by center
    # https://stackoverflow.com/questions/16646183/crop-an-image-in-the-centre-using-pil
    left = (img.size[0] - 224) / 2
    bottom = (img.size[1] - 224) / 2
    right = (img.size[0] + 224) / 2
    top = (img.size[1] + 224) / 2

    # Crop the center of the image
    crop_img = img.crop((left, bottom, right, top))
    print(crop_img.size)
    # np_image
    np_image = np.array(crop_img) / 255.0
    print(np.array(crop_img).shape)

    # Normalization
    means = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    norm_image = (np_image - means) / std
    tmp = np_image - means

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


def predict(image_path, model, topk):
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
def plot_predicts(image_path, cat_names_path, model, topk):
    cat_to_name = label_mapping(cat_names_path)
    img = process_image(image_path)
    probs, classes = predict(image_path, model, topk)
    names = [cat_to_name[i] for i in classes]

    # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.figure.html
    plt.figure(figsize=(4, 10))
    # https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.subplot.html
    ax = plt.subplot(211)

    plt.axis(
        'off')  # https://stackoverflow.com/questions/9295026/matplotlib-plots-removing-axis-legends-and-white-spaces
    title = ax.set_title(names[0])
    imshow(img, ax, title)

    plt.subplot(212)
    # https://seaborn.pydata.org/generated/seaborn.barplot.html
    # https://seaborn.pydata.org/tutorial/color_palettes.html
    sns.barplot(x=probs, y=names, color=sns.color_palette("Blues")[-2])

    # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots_adjust.html
    plt.subplots_adjust(left=0.125, right=0.9, bottom=0.3, top=1.0, hspace=0.05)
    plt.show()