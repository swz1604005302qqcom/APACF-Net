import torch
import numpy as np
import os
import random
import copy
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


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


def data_normal_2d(orign_data):
    dim = 0
    d_min = torch.min(orign_data,dim=dim)[0]
    d_min = d_min.detach().cpu().numpy()
    orign_data = orign_data.detach().cpu().numpy()
    for idx,j in enumerate(d_min):  # 就是这句话，导致loss 无法反向传播
        if j < 0:
            orign_data[idx,:] = orign_data[idx,:] + np.abs(d_min[idx])
            d_min = orign_data.min(0)

    orign_data = torch.tensor(orign_data)
    d_min = torch.tensor(d_min)
    d_max = torch.max(orign_data,dim=dim)[0]

    dst = d_max - d_min
    if d_min.shape[0] == orign_data.shape[0]:
        d_min = d_min.unsqueeze(1)
        dst = dst.unsqueeze(1)
    else:
        d_min = d_min.unsqueeze(0)
        dst = dst.unsqueeze(0)
    norm_data = torch.sub(orign_data,d_min).true_divide(dst)
    norm_data = (norm_data - 0.5).true_divide(0.5)
    return norm_data




if __name__ == '__main__':
    batch = 20
    logit = torch.randn(batch, 8)
    truth = []
    for k in range(batch):
        d = random.randint(0, 1)
        truth.append(d)
    truth = torch.tensor(truth)
    distAll = IntraClass_Loss(logit, truth)

    print(distAll)