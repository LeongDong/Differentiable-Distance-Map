import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES']="1"
from ReadDisMark import readDisMark
import cv2
savepath = ''
imgname = '97.png'
path = '/home/liang/Data/txt/DistanceMask.mat'
truePath = '/home/liang/Data/DistanceMap/mask/'
predPath = '/home/liang/Data/DistanceMap/mask/'

def soft_erode(img):

    p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
    p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
    return torch.min(p1, p2)

def boundMask(img):

    img_clone = img.clone()
    img_clone = img_clone.detach()
    inner = soft_erode(img_clone)
    return img - inner

def disCal(img, kernel, maxValue = 10, beta = -2): #img:N*C*H*W; kernel:1*1*H*W

    N, C, H, W = img.shape
    distmap = torch.zeros_like(img)
    _, __, KH, KW = kernel.shape #1*1*KH*KW
    # bounprob = boundMask(img)
    bounprob = img
    imgpad = F.pad(bounprob, (W - 1, W - 1, H - 1, H - 1), 'constant', 0)
    imgpad = 1.1 / (imgpad + 0.1)
    for i in range(H):
        for j in range(W):
            imgblock = imgpad[0, 0, i:(i + 2 * H - 1), j:(j + 2 * W - 1)]
            imgmin = imgblock * kernel * beta
            distmap[0,0,i,j] = 1 / beta * torch.log(torch.sum(torch.exp(imgmin)))

    return distmap


if __name__ == "__main__":

    trueimg = cv2.imread(truePath + imgname)
    predimg = cv2.imread(predPath + imgname)

    trueimg = cv2.cvtColor(trueimg, cv2.COLOR_BGR2GRAY)
    predimg = cv2.cvtColor(predimg, cv2.COLOR_BGR2GRAY)

    trueimg = cv2.resize(trueimg, (256, 256), interpolation=cv2.INTER_LINEAR)
    predimg = cv2.resize(predimg, (256, 256), interpolation=cv2.INTER_LINEAR)
    trueimg = trueimg / 255
    predimg = predimg / 255
    binError = np.mean(np.abs(trueimg - predimg))
    diceError = 1 - 2 * np.sum(trueimg * predimg) / (np.sum(trueimg) + np.sum(predimg))

    trueimg = torch.from_numpy(trueimg)
    predimg = torch.from_numpy(predimg)
    trueimg = trueimg.cuda()
    predimg = predimg.cuda()
    H, W = trueimg.shape

    trueimg = trueimg.unsqueeze(0).unsqueeze(0)
    predimg = predimg.unsqueeze(0).unsqueeze(0)

    kernel = readDisMark(path)
    kernel = kernel + 1
    kernel = np.log2(kernel) + 1
    kernel = torch.from_numpy(kernel)
    kernel = kernel.cuda()
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    predDisMap = disCal(predimg, kernel)
    trueDisMap = disCal(trueimg, kernel)

    predDisMap = torch.squeeze(predDisMap)
    predDisMap = predDisMap.cpu().numpy()
    trueDisMap = torch.squeeze(trueDisMap)
    trueDisMap = trueDisMap.cpu().numpy()
    Error = np.mean(abs(predDisMap - trueDisMap))
    print('Dice loss is:{}, Bin L1 loss is:{}, L1 loss is:{}'.format(diceError, binError, Error))

    x = []
    y = []
    for i in range(W):
        x.append(i + 1)
    for i in range(H):
        y.append(H - i)
    x = np.array([x])
    y = np.array([y]).T
    xc = np.repeat(x, H, axis=0)
    yc = np.repeat(y, W, axis=1)
    plt.figure(figsize=(10,10))
    plt.rcParams['figure.figsize'] = [3,3]
    # plt.subplot(2,1,1)
    heatmap = plt.pcolormesh(xc, yc, predDisMap, cmap='viridis')#, shading='gouraud')
    # plt.subplot(2,1,2)
    # heatmap2 = plt.pcolormesh(xc, yc, trueDisMap, cmap='viridis', shading='gouraud')
    plt.colorbar(heatmap, label='AUPR')
    plt.tight_layout()
    plt.show()
# import torch
# import numpy as np
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import os
# os.environ['CUDA_VISIBLE_DEVICES']="1"
# import cv2
# savepath = '/home/liang/Data/DistaneMap/pred/97.png'
# imgname = '97.png'
# Path = '/home/liang/Data/DistaneMap/mask/'
#
# def soft_erode(img):
#
#     p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
#     p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
#     return torch.min(p1, p2)
#
# def boundMask(img):
#     inner = soft_erode(img)
#     inn_area = inner.clone()
#     inn_area = inn_area.detach()
#
#     return img - inn_area
#
# if __name__ == "__main__":
#
#     img = cv2.imread(Path + imgname)
#
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = img / 255
#     img = torch.from_numpy(img)
#     img = img.cuda()
#     img = img.unsqueeze(0).unsqueeze(0)
#     bound = boundMask(img)
#     bound = torch.squeeze(bound)
#     bound = bound.cpu().numpy()
#     bound = bound * 255
#     bound = bound.astype('uint8')
#     cv2.imwrite(savepath, bound)
