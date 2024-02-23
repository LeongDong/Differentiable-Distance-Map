import scipy.io as sio
# path = '/home/liang/Data/txt/DistaceMask.mat'
def readDisMark(path):

    data = sio.loadmat(path)
    disMark = data['distanceMark']

    return disMark