import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', '-m', default='./checkpoints_psp_split/split1_1.25/CP_epoch10.pth', metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', type=str, default='/home/leognha/Desktop/Advanced-BicycleGAN-mask/PPT/compare/',
                        help='filenames of input images', required=False)
    parser.add_argument('--mask', '-mask', type=str, default='./data/masks/',
                        help='mask of input images', required=False)                   
    parser.add_argument('--prob', '-p', type=str, help='Save prob path and name', default='./Prob/')
    parser.add_argument('--output', '-o', type=str, default='./output/',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1)

    return parser.parse_args()
