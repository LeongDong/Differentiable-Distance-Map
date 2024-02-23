import numpy as np
import scipy.io as sio
H = 144
W = 144
KH = 2 * H - 1
KW = 2 * W - 1
KH_2 = np.ceil(KH / 2)
KW_2 = np.ceil(KW / 2)
kernel = np.zeros((KH, KW), dtype=np.float64)
path = '/home/liang/Data/txt/DistanceMask.mat'
if __name__ == "__main__":

    for i in range(KH):
        for j in range(KW):
            kernel[i,j] = np.sqrt((i + 1 - KH_2) * (i + 1 - KH_2) + (j + 1 - KW_2) * (j + 1 - KW_2))
    data = {'distanceMark': kernel}
    sio.savemat(path, data)

    # data = sio.loadmat(path)
    # disMark = data['distanceMark']
    # print(disMark)