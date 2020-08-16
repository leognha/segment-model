import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff

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
    
    if intersections ==0 or union==0:
        overlaps = 0
    else:
        overlaps = intersections / union

    return overlaps

def miou_loss(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.num_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    #batch_size = loader.batch_size 
    tot = 0
    print('')
    with tqdm(total=n_val, desc='mIOU round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                # SegNet
                # mask_pred, mask_softmax = net(imgs)
                # DeepLabv3+
                mask_pred = net(imgs)

            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()

            for i in range(loader.batch_size):
                tot += compute_overlaps_masks(pred[i], true_masks[i])
                #print(compute_overlaps_masks(pred[i], true_masks[i]))
            pbar.update()
            
    return tot / (n_val*loader.batch_size)
