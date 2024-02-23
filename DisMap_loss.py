import torch
import torch.nn as nn
import torch.nn.functional as F

def soft_erode(img):

    p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
    p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
    return torch.min(p1, p2)

def boundMask(img, mask):

    inner = soft_erode(mask)
    return img - inner, mask - inner

def disCal(img, mask, kernel):

    N, C, H, W = mask.shape
    distmap_gt = torch.zeros_like(mask)
    distmap_pd = torch.zeros_like(mask)
    _, __, KH, KW = kernel.shape
    maskpad = F.pad(mask, (W - 1, W - 1, H - 1, H - 1), 'constant', 0)
    imgpad = F.pad(img, (W - 1, W - 1, H - 1, H - 1), 'constant', 0)
    maskpad = 1.1 / (maskpad + 0.1)
    kernel_flat = kernel.view(-1)
    for i in range(H):
        for j in range(W):
            imgblock = imgpad[0, 0, i:(i + 2 * H - 1), j:(j + 2 * W -1)]
            maskblock = maskpad[0, 0, i:(i + 2 * H - 1), j:(j + 2 * W -1)]
            img_flat = imgblock.contiguous().view(-1)
            mask_flat = maskblock.contiguous().view(-1)
            maskmin = mask_flat * kernel_flat
            min_value, min_loc = torch.min(maskmin) #1 / beta * torch.log(torch.sum(torch.exp(bounmin)))
            distmap_gt[0, 0, i, j] = min_value
            distmap_pd[0, 0, i, j] = img_flat[min_loc] * kernel_flat[min_loc]

    return distmap_pd, distmap_gt

class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, pred, mask, kernel):

        pred, mask = boundMask(pred, mask)
        loss = torch.sum((pred - mask) * (pred - mask))
        return loss