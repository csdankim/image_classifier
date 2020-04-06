import argparse

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

from image_classifier import load_data, train, validation, build_classifier, save_model


def main():
    parser = argparse.ArgumentParser(description='Train image classifier.')

    parser.add_argument('--data_directory', action='store',
                        dest='data_directory', default='../aipnd-project/flowers',
                        help='Enter the training data path')

    parser.add_argument('--arch', action='store',
                        dest='pretrained_model', default='vgg16',
                        help='Enter the pretrained model to use. \
                              The default is VGG16. \
                              Be aware of the number of input units.')

    parser.add_argument('--save_dir', action='store',
                        dest='save_directory', default='checkpoint.pth',
                        help='Enter the path to save checkpoint.pth.')

    parser.add_argument('--learning_rate', action='store',
                        dest='lr', type=float, default=0.001,
                        help='Enter the learning rate for the training model. \
                              The default learning rate is 0.001.')

    parser.add_argument('--dropout', action='store',
                        dest='dropout', type=float, default=0.5,
                        help='Enter the dropout rate for the training model. \
                              The default dropout rate is 0.05.')

    # parser.add_argument('--input_units', action='store',
    #                     dest='input_units', type=int, default=25088,
    #                     help='Enter the number of input units in the classifier. \
    #                           The default input unit size is 25088 because the default model is VGG16. \
    #                           Must be aware of the input unit size your chosen model.')

    parser.add_argument('--hidden_units', action='store',
                        dest='hidden_units', type=int, default=4096,
                        help='Enter the number of hidden units in the classifier. \
                              The default hidden units size is 4096.')

    parser.add_argument('--epochs', action='store',
                        dest='epochs', type=int, default=15,
                        help='Enter the number of epochs for the training model. \
                              The default is 15')

    parser.add_argument('--print_every', action='store',
                        dest='print_every', type=int, default=40,
                        help='Enter the number of print round for the training model. \
                              The default is 40')

    parser.add_argument('--gpu', action='store_true',
                        dest='gpu', default=False,
                        help='Turn GPU mode on or off. The default mode is CPU.')

    parser.add_argument('--version', action='version',
                        version='%(prog)s 1.0')

    results = parser.parse_args()

    data_dir = results.data_directory
    pretrained_model = results.pretrained_model
    save_dir = results.save_directory
    learning_rate = results.lr
    dropout = results.dropout
    # input_units = results.input_units
    hidden_units = results.hidden_units
    epochs = results.epochs
    print_every = results.print_every
    gpu_mode = results.gpu

    print('data_dir         = {!r}'.format(data_dir))
    print('pretrained_model = {!r}'.format(pretrained_model))
    print('save_dir         = {!r}'.format(save_dir))
    print('learning_rate    = {!r}'.format(learning_rate))
    print('dropout_rate     = {!r}'.format(dropout))
    print('hidden_units     = {!r}'.format(hidden_units))
    print('number of epochs = {!r}'.format(epochs))
    print('print_round      = {!r}'.format(print_every))
    print('gpu_mode         = {!r}'.format(gpu_mode))

    # Load and preprocess data
    train_data, valid_data, test_data, trainloader, validloader, testloader = load_data(data_dir)

    # Load pre-trained model
    model = getattr(models, pretrained_model)(pretrained=True)

    # build classifier
    input_units = model.classifier[0].in_features  # model classifier's input size
    model = build_classifier(model, input_units, hidden_units, dropout)

    # Use GPU if it's available
    device = torch.device("cuda" if gpu_mode else "cpu")
    model.to(device)
    print(model)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

    print("\nstart training")
    train_loss, valid_loss, valid_accuracy = train(model, device, criterion, optimizer, trainloader, validloader, epochs,
                                                   print_every)

    save_model(model, train_data, optimizer, save_dir, epochs)

    print("\nstart testing")
    test_loss, test_accuracy = validation(model, device, criterion, testloader)
    print(f"test loss: {test_loss:.3f}.. "
          f"test accuracy: {test_accuracy:.3f}")


if __name__ == '__main__':
    main()