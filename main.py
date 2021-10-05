import torch
from util.load_data import *

if __name__ == '__main__':
    train_file = '/Users/yu/PycharmProjects/NGCF/NGCF-PyTorch/Data/gowalla/train.txt'
    test_file = '/Users/yu/PycharmProjects/NGCF/NGCF-PyTorch/Data/gowalla/test.txt'
    data = Data(train_file, test_file)

    data.read_dataset()

    data.create_adj_mat()
    data.get_adj_mat()