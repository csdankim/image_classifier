import argparse
from torchvision import datasets, transforms, models
from image_classifier import load_checkpoint, plot_predicts

def main():
    parser = argparse.ArgumentParser(description='Predict Image Class.')

    parser.add_argument('--image_path', action='store',
                        dest='image_path', default='../aipnd-project/flowers/test/11/image_03141.jpg',
                        help='Enter the image path')

    parser.add_argument('--arch', action='store',
                        dest='pretrained_model', default='vgg16',
                        help='Enter the pretrained model to use. \
                              The default is VGG16. \
                              Be aware of the number of input units.')

    parser.add_argument('--save_dir', action='store',
                        dest='save_directory', default='checkpoint.pth',
                        help='Enter the path to the saved model. \
                              The default is checkpoint.pth')

    parser.add_argument('--top_k', action='store',
                        dest='top_k', type=int, default=3,
                        help='Enter the number of the most likely classes to view. \
                              The default top_k is 3.')

    parser.add_argument('--category_names', action='store',
                        dest='cat_to_name_path', default='cat_to_name.json',
                        help='Enter the path to the category classification labels. \
                              The default is cat_to_name.json')

    parser.add_argument('--gpu', action='store_true',
                        dest='gpu', default=False,
                        help='Turn GPU mode on or off. The default mode is CPU.')

    parser.add_argument('--version', action='version',
                        version='%(prog)s 1.0')

    results = parser.parse_args()

    image_path = results.image_path
    pretrained_model = results.pretrained_model
    save_dir = results.save_directory
    topk = results.top_k
    cat_names_path = results.cat_to_name_path
    gpu_mode = results.gpu

    print('image_path       = {!r}'.format(image_path))
    print('pretrained_model = {!r}'.format(pretrained_model))
    print('saved_model_dir  = {!r}'.format(save_dir))
    print('top_k            = {!r}'.format(topk))
    print('cat_label_path   = {!r}'.format(cat_names_path))
    print('gpu_mode         = {!r}'.format(gpu_mode))

    # Load pre-trained model
    model = getattr(models, pretrained_model)(pretrained=True)

    model, _, _ = load_checkpoint(model, save_dir, gpu_mode)
    plot_predicts(image_path, cat_names_path, model, topk)


if __name__ == '__main__':
    main()
    