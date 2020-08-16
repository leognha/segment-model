import logging
import os

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from glob import glob

import segmentation_models_pytorch as smp

# from CENet.cenet import CE_Net_
# from unet import UNet

#from DeepLabv3plus.deeplab import *
from SegNet.Segnet import SegNet

from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
from utils.test_segnet_utils import get_args

args = get_args()

model_path = args.model
prob_path = args.prob

def predict_img(net,
                full_img,
                device,
                scale_factor=0.427,
                out_threshold=0.5,
                fn=None,
                args=None):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, 0.427))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32) 
    


    with torch.no_grad():

        # SegNet
        output, output_softmax = net(img)

        # DeepLabv3+
        # output = net(img)

        if net.num_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)
        '''
        save_array = probs.cpu().numpy()
        mat_prob=np.reshape(save_array, [300, 300])
        #print("mat: {}".format(mat_prob))
        save_path = save_mats_path()
        save_fn = save_path+fn.split('.')[0]+'.mat'
        sio.savemat(save_fn, {'array' : mat_prob})
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(448),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        '''
        a= probs.squeeze()
        full_mask = probs.squeeze().cpu().numpy()
        

        a[a > out_threshold] = 1
        # 矩阵a中小鱼Threshold(阈值)的部分置0
        a[a < out_threshold] = 0

    return full_mask > out_threshold, a


def get_classes_name():
    classes_name = ""
    net = model_path.split('/')[1]
    model = model_path.split('/')[2]
    model = model.split('.')[0]
    print("net = {}, model = {}".format(net, model))
    classes_name = net + '_' + model + '/'
    return classes_name


def save_mats_path():
    classes_name = get_classes_name() 
    prob_files_path = prob_path + classes_name
    if not os.path.exists(prob_files_path):
            os.mkdir(prob_files_path)
    return prob_files_path

def get_files_name():
    dst = []
    input_files_path = args.input
    tmp = os.listdir(input_files_path)
    for i in tmp:
        dst.append(i.split('.')[0])
    #print("files_path: {}".format(dst[0]))
    return dst

def get_input_data(files_path):
    dst = []
    #mask_dst = []
    input_path = args.input
    mask_path = args.mask

    for fn in files_path:
        dst.append(input_path + fn + ".png")
        #dst.append(input_path + fn + ".jpg")
        #mask_dst.append(mask_path + fn + ".png")
    #print("input data img path: {}".format(dst[0]))


    return dst #,mask_dst



def save_output_path(files_path):
    dst = []
    output_path = args.output
    classes_name = get_classes_name()
    output_path = output_path + classes_name
    if not os.path.exists(output_path):
            os.mkdir(output_path)
    for fn in files_path:
        dst.append(output_path + fn + ".png")
    print("save output imgs path: {}".format(dst[0]))
    return dst

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def compute_overlaps_masks(masks1, masks2):
    '''
    masks1, masks2: [Height, Width, instances]
    this for binary mask
    '''
    
    # If either set of masks is empty return empty result
    if masks1.shape[0] == 0 or masks2.shape[0] == 0:
        return np.zeros((masks1.shape[0], masks2.shape[-1]))
    # flatten masks and compute their areas
    #masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    #masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    #area1 = np.sum(masks1, axis=0)
    #area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = torch.sum(torch.mul(masks1, masks2)) #correct part
    union = torch.sum((masks1 + masks2)-torch.mul(masks1, masks2))
    overlaps = intersections / union
    if intersections ==union:
        overlaps=1
    return overlaps


if __name__ == "__main__":
    filenames = get_files_name()
    #in_files,mask_files = get_input_data(filenames)
    in_files = get_input_data(filenames)

    
    #print("in_files = {}".format(in_files))
    out_files = save_output_path(filenames)

    # net = UNet(n_channels=3, n_classes=1)
    # net = CE_Net_(n_classes=1, n_channels=3)
    net = SegNet(num_classes=1, input_channels=3, output_channels=1)

    # Define network
    '''
    net = DeepLab(num_classes=1,
                 backbone='resnet',
                  output_stride=16,
                  sync_bn=None,
                  freeze_bn=False,
                  input_channels=3)
   
    '''
    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    mIOU = 0
    i_numer =0

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)
        '''
        name = fn.split('/')[-1]
        camera_idx = name.split('_')[0]
        mask_files = glob(args.mask + camera_idx + '.png')
        mask = Image.open(mask_files[0])
        masks=mask.convert('L')

        mask_gt = torch.from_numpy(BasicDataset.mask_preprocess(1,masks, 128, 128))

        #mask_gt = mask_gt.unsqueeze(0)
        
        mask_gt = mask_gt.to(device=device, dtype=torch.float32) 
        mask_gt = mask_gt.squeeze()#.cpu().numpy()
        '''
        img=img.convert(mode='RGB')



        mask,binary_mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device,
                           fn=fn.split('/')[-1], 
                           args=args)

        #Iou = compute_overlaps_masks(binary_mask,mask_gt)
        #mIOU = ((mIOU*i) + Iou)/(i+1) 
        #mIOU = mIOU +Iou
        print(i)
        #print(mIOU)
        i_numer = i
        
        #print(mIOU/(i_numer + 1))
        if not args.no_save:
            out_fn = out_files[i]
            result = mask_to_image(mask)
            #print("mask {}:".format(mask))
            #cv2.imwrite(out_files[i], mask)
            result.save(out_files[i])

            logging.info("Mask saved to {}".format(out_files[i]))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)
        
