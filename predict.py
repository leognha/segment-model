import logging
import os

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
# from CENet.cenet import CE_Net_
# from unet import UNet
#from DeepLabv3plus.deeplab import *
from SegNet.Segnet import SegNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
from utils.predict_utils import get_args


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    #print(img.shape)
    #img = torch.tensor(full_img, device=device).float()
    #img = img.unsqueeze(0)

    with torch.no_grad():
        # SegNet
        output, output_softmax = net(img)
        # DeepLabv3+
        #output = net(img)

        if net.num_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)
        save_array = probs.cpu().numpy()
        mat_prob=np.reshape(save_array, [300, 300])
        print("mat: {}".format(mat_prob))
        save_fn = '/home/leognha/Desktop/seg-model/MedicalImage_Project02_Segmentation/Prob/prob.mat'
        sio.savemat(save_fn, {'array' : mat_prob})
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(448),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    # net = UNet(n_channels=3, n_classes=1)
    # net = CE_Net_(n_classes=1, n_channels=3)
    # net = SegNet(num_classes=1, input_channels=3, output_channels=1)

    # Define network
    net = DeepLab(num_classes=1,
                  backbone='resnet',
                  output_stride=16,
                  sync_bn=None,
                  freeze_bn=False,
                  input_channels=3)


    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)
        img=img.convert(mode='RGB')


        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)


        if not args.no_save:
            out_fn = out_files[i]
            result = mask_to_image(mask)
            print("mask {}:".format(mask))

            result.save(out_files[i])

            logging.info("Mask saved to {}".format(out_files[i]))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)
