import torch
from util.load_data import *
from util.parser import parse_args
from NGCF.NGCF import NGCF


if __name__ == '__main__':
    args = parse_args()
    train_file = args.data_path + '/' + args.dataset + '/' + args.train_file
    test_file = args.data_path + '/' + args.dataset + '/' + args.test_file

    data = Data(train_file, test_file)
    data.read_dataset()

    data.create_adj_mat()
    norm_adj = data.get_adj_mat()

    args.node_dropout = eval(args.node_dropout)
    args.message_dropout = eval(args.message_dropout)

    model = NGCF(data.n_users, data.n_items, norm_adj, args)

    for epoch in range(args.epoch):
        print(epoch)
        n_batch = data.n_train_ratings // args.batch_size + 1

        for idx in range(n_batch):
            print(idx)