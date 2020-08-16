import argparse
import logging
import os
import sys

import torch
import torch.nn as nn

from torch import optim
from torchvision import transforms
from tqdm import tqdm

from eval_segnet import eval_net
from miou_segnet import miou_loss

# from unet import UNet
# from CENet.cenet import CE_Net_
# from DeepLabv3plus.deeplab import *
from SegNet.Segnet import SegNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
# from utils.ce_dataset import ImageFolder
from utils.train_utils import get_args
from torch.utils.data import DataLoader, random_split

#dir_img = 'data_origin/imgs/'
#dir_mask = 'data_origin/masks/'
dir_img = 'data/imgs/'
dir_mask = 'data/raw_mask/'
dir_img_agu = 'data/agu_17w/'

dir_checkpoint = 'checkpoints_seg_17w/'


def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=1):
    
    dataset = BasicDataset(dir_img, dir_mask, dir_img_agu,img_scale)


    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.num_classes > 1 else 'max', patience=2)
    if net.num_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    f = open(dir_checkpoint+'miou.txt','w')

    for epoch in range(2,epochs):

        #miou_score = miou_loss(net, val_loader, device)
        #print('miou_score=',miou_score)
        #break

        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                # print(imgs.shape)
                true_masks = batch['mask']
                assert imgs.shape[1] == net.input_channels, \
                    f'Network has been defined with {net.input_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                #print("images shape: {}".format(imgs.shape))
                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.num_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                
                # SegNet
                masks_pred, masks_softmax = net(imgs)
                # DeepLabv3+
                #masks_pred = net(imgs)

                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
            if epoch != 0:
                     #for tag, value in net.named_parameters():
                     #   tag = tag.replace('.', '/')
                     #   writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                     #   writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                val_score = eval_net(net, val_loader, device)
                scheduler.step(val_score)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                if net.num_classes > 1:
                    logging.info('Validation cross entropy: {}'.format(val_score))
                    writer.add_scalar('Loss/test', val_score, global_step)
                else:
                    logging.info('Validation Dice Coeff: {}'.format(val_score))
                    writer.add_scalar('Dice/test', val_score, global_step)

                writer.add_images('images', imgs, global_step)
                if net.num_classes == 1:
                    writer.add_images('masks/true', true_masks, global_step)
                    writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
            
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
        
            miou_score = miou_loss(net, val_loader, device)
            print('miou_score=',miou_score)
            print('miou_score=',miou_score,file = f)
    f.close
    writer.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print("--------------------CUDA is available------------------")
        device = torch.device('cuda:5')
    else:
        device = torch.device('cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # input_channels=3 for RGB images
    # num_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use num_classes=1
    #   - For 2 classes, use num_classes=1
    #   - For N > 2 classes, use num_classes=N
    # net = UNet(input_channels=3, num_classes=1, bilinear=True)
    # net = CE_Net_(num_classes=1, input_channels=3)
    # net = resnet.resnet50(pretrained=True)
    net = SegNet(num_classes=1, input_channels=3, output_channels=1)
    # net = DeepLab(num_classes=1, backbone='resnet', output_stride=16, sync_bn=None, freeze_bn=False, input_channels=3)
    # Define network
    
    
    logging.info(f'Network:\n'
                 f'\t{net.input_channels} input channels\n'
                 f'\t{net.num_classes} output channels (classes)\n'
                 )

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
            
        )
        logging.info(f'Model loaded from {args.load}')
    # net.to(device=device)
    net.cuda(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
