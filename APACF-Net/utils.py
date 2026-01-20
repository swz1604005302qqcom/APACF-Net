from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import time
from timeit import default_timer as timer
from torch.utils.data.sampler import *
import torch.nn.functional as F
import os
import shutil
import sys
import numpy as np
import torch

def save(list_or_dict,name):
    f = open(name, 'w')
    f.write(str(list_or_dict))
    f.close()

def load(name):
    f = open(name, 'r')
    a = f.read()
    tmp = eval(a)
    f.close()
    return tmp

def acc(preds,targs,th=0.0):
    preds = (preds > th).int()
    targs = targs.int()
    return (preds==targs).float().mean()

def dot_numpy(vector1 , vector2,emb_size = 512):
    vector1 = vector1.reshape([-1, emb_size])
    vector2 = vector2.reshape([-1, emb_size])
    vector2 = vector2.transpose(1,0)
    cosV12 = np.dot(vector1, vector2)
    return cosV12

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

#交叉熵损失，这是多类别分类问题的标准损失函数。
def softmax_cross_entropy_criterion(logit, truth, is_average=True):
    loss = F.cross_entropy(logit, truth, reduce=is_average)
    return loss
#二元交叉熵损失，适用于二分类问题。
def bce_criterion(logit, truth, is_average=True):
    loss = F.binary_cross_entropy_with_logits(logit, truth.view(-1,1), reduce=is_average)
    return loss

def IntraClass_Loss(logit, truth):
    dim = truth.shape
    logitNorm = data_normal_2d(logit)
    logitNorm = logitNorm.detach().cpu().numpy()
    posIdx, negIdx = [], []
    for idx in range(dim[0]):
        gtLabel = truth[idx].detach().cpu().numpy()
        if gtLabel == 1:
            posIdx.append(idx)
        else:
            negIdx.append(idx)
    # 计算正样本对之间的欧式距离
    distPos = np.zeros((1,))
    numP = len(posIdx)
    iterP = (numP * (numP - 1))/2
    while len(posIdx):
        posFeature1 = logitNorm[posIdx[0]]
        posTmp = copy.deepcopy(posIdx) #复制列表里的值
        posTmp.pop(0) #删除列表里的第一个值，接下来进行循环，第一个和第二个比较，循环以此类推
        for j in posTmp:
            posFeature2 = logitNorm[j]
            dist = np.sqrt(np.sum((posFeature1 - posFeature2) ** 2))
            distPos = distPos + dist
        posIdx = copy.deepcopy(posTmp)
    distPos = distPos/iterP

    # 计算负样本对之间的欧式距离
    distNeg = np.zeros((1,))
    numN = len(negIdx)
    iterN = (numN * (numN - 1))/2
    while len(negIdx):
        negFeature1 = logitNorm[negIdx[0]]
        negTmp = copy.deepcopy(negIdx)
        negTmp.pop(0)
        for j in negTmp:
            negFeature2 = logitNorm[j]
            dist = np.sqrt(np.sum((negFeature1 - negFeature2) ** 2))
            distNeg = distNeg + dist
        negIdx = copy.deepcopy(negTmp)
    distNeg = distNeg/iterN
    distAll = distPos + distNeg
    distAll = torch.tensor(distAll, dtype=torch.float32)
    return distAll

def remove_comments(lines, token='#'):
    """ Generator. Strips comments and whitespace from input lines.
    """
    l = []
    for line in lines:
        s = line.split(token, 1)[0].strip()
        if s != '':
            l.append(s)
    return l

def remove(file):
    if os.path.exists(file): os.remove(file)

def empty(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir, ignore_errors=True)
    else:
        os.makedirs(dir)

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1 ):
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)
    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)
    else:
        raise NotImplementedError

def np_float32_to_uint8(x, scale=255.0):
    return (x*scale).astype(np.uint8)

def np_uint8_to_float32(x, scale=255.0):
    return (x/scale).astype(np.float32)

