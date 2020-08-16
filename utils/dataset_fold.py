from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, imgs_agu_dir, scale=1, transform=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.imgs_agu_dir = imgs_agu_dir
        self.scale = scale
        self.transform = transform
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]

        self.ids_agu = [splitext(file)[0] for file in listdir(imgs_agu_dir)
                    if not file.startswith('.')]
        self.origun_data_number = len(self.ids)
        self.ids = self.ids + self.ids_agu

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans.astype(np.float64)

    def mask_preprocess(cls, pil_img, w,h):
        #w, h = pil_img.size
        #newW, newH = int(scale * w), int(scale * h)
        #assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((w, h))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans.astype(np.float64)

    def __getitem__(self, i):
        idx = self.ids[i]
        camera_idx = idx.split('_')[0]
        #print(camera_idx)

        #mask_file = glob(self.masks_dir + idx + '*')
        mask_files = glob(self.masks_dir + camera_idx + '.png')

        if i < self.origun_data_number:
            img_file = glob(self.imgs_dir + idx + '*')
        else:
            img_file = glob(self.imgs_agu_dir + idx + '*')

        #assert len(mask_file) == 1, \
        #    f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        # print("load img: {}".format(img_file[0]))
        # print("load mask: {}".format(mask_file[0]))

        #try:
        #    mask = Image.open(mask_files[0])
        #except:
        #    print(mask_files)
        #    print(len(mask_files))
        
        mask = Image.open(mask_files[0])
        

        img = Image.open(img_file[0])
        # mask.resize((224, 224))
        # img.resize((224, 224))
        #print(mask)
        masks=mask.convert('L')
        #print(mask)
        img=img.convert(mode='RGB')

        #assert img.size == mask.size, \
        #    f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        #img = self.preprocess(img, 128)
        img = self.mask_preprocess(img,128,128)
        if self.transform:
            img = self.transform(img)

        
        maskss = self.mask_preprocess(masks,128,128)

        #print(torch.sum(torch.from_numpy(mask)))

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(maskss)}
