import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict


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
    
    return trainloader, validloader, testloader


def validation(dataloader):
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
            
   return valid_loss/len(validloader), accuracy/len(validloader)


def train(trainloader, validloader, epochs=15, print_every=40):
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
                valid_loss, valid_accuracy = validation(validloader)
                
                train_loss = running_loss / print_every

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {train_loss:.3f}.. "
                      f"Valid loss: {valid_loss:.3f}.. "
                      f"Valid accuracy: {valid_accuracy:.3f}")
                running_loss = 0
                model.train()
                
    # TODO: Save the checkpoint 
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'class_map': model.class_to_idx,
                  'optimizer': optimizer.state_dict,
                  'classifier': model.classifier,
                  'epochs': epochs,
                  'state_dict': model.state_dict()}
    
    torch.save(checkpoint, 'checkpoint.pth')
    
    return train_loss, valid_loss, valid_accuracy
    
    
def main():
    trainloader, validloader, testloader = load_data('flower')
    
    # TODO: Build and train your network
    model = models.vgg16(pretrained=True)
    
    # Freeze only features parameters so we don't backprop through them
    # instead of making new classifier, I would like to use the same structure as VGG16, 
    # but I just want to train classifier only. 
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, 4096, bias=True)),
                              ('relu1', nn.ReLU()),
                              ('dropout1', nn.Dropout(p=0.5)),
                              ('fc2', nn.Linear(4096, 128, bias=True)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    
    train_loss, valid_loss, valid_accuracy = train(trainloader, validloader, epochs=15, print_every=40)
    
    test_loss, test_accuracy = validation(testloader)
    print(f"test loss: {test_loss:.3f}.. "
          f"test accuracy: {test_accuracy:.3f}")
    

if __name__ == '__main__':
    main()
    
    
