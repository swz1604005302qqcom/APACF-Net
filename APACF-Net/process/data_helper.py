import os
import random
from utils import *

DATA_ROOT1 = r'/root/autodl-tmp/CASIA-SURF/phase1'
#DATA_ROOT1 = r'/root/autodl-tmp/CASIA-SURF-CeFA/phase1'
TRN_IMGS_DIR = DATA_ROOT1 + '/Training/'
TST_IMGS_DIR = DATA_ROOT1 + '/Val/'
RESIZE_SIZE = 112

def load_train_list():
    list = []
    f = open(DATA_ROOT1 + '/train_list.txt')
    #f = open(DATA_ROOT1 + '/4@2_train.txt')
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)
    return list

def load_val_list():
    list = []
    #f = open(DATA_ROOT1 + '/CeFA_dev_NoPca_3.txt')
    f = open(DATA_ROOT1 + '/val_private_list.txt')
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)
    return list

def load_test_list():
    list = []
    f = open(DATA_ROOT1 + '/test_private_list.txt')
    #f = open(DATA_ROOT + '/4_1_test.txt')
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)

    return list

# def load_test_list():
#     with open(DATA_ROOT1 + '/4_1_test.txt') as f:
#         lines = f.readlines()
#     print(f"Total lines in 4_1_test.txt: {len(lines)}")
#     data_list = [line.strip().split(' ') for line in lines]
#     return data_list

def transform_balance(train_list):
    pos_list = []
    neg_list = []
    for tmp in train_list:
        if tmp[3]=='1':
            pos_list.append(tmp)
        else:
            neg_list.append(tmp)

    print(len(pos_list))
    print(len(neg_list))
    return [pos_list,neg_list]

def submission(probs, outname, mode='valid'):
    if mode == 'valid':
        f = open(DATA_ROOT + '/val_public_list.txt')
    else:
        f = open(DATA_ROOT + '/test_public_list.txt')

    lines = f.readlines()
    f.close()
    lines = [tmp.strip() for tmp in lines]

    f = open(outname,'w')
    for line,prob in zip(lines, probs):
        out = line + ' ' + str(prob)
        f.write(out+'\n')
    f.close()
    return list



