import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Train the DeepLab3++ on images and target masks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-bc', '--binary_class', type=int, default=1, help='BINARY_CLASS', dest='BINARY_CLASS')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=10,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=5,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0002,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.427,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-c', '--cuda devices', dest='val', type=int, default=1,
                        help='The cuda device id')

    return parser.parse_args()
