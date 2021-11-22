import torch
from util.load_data import *
from util.parser import parse_args
from NGCF.NGCF import NGCF
from MPT.train import Trainer
import torch.optim as optim
from time import time
from util.test import *


def train():

    model = NGCF(data.n_users, data.n_items, norm_adj, args).to('cuda')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epoch):
        t0_start = time()
        loss = 0
        n_batch = data.n_train_ratings // args.batch_size + 1

        for idx in range(n_batch):
            users, pos_items, neg_items = data.sample()

            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users, pos_items, neg_items, drop_flag=args.node_dropout)

            batch_loss = model.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss

        t0_end = time()
        print('epoch {} : loss {} , time {}s'.format(epoch + 1, loss.item(), t0_end - t0_start))

        if (epoch+1) % 100 == 0:
            users_to_test = list(data.test_set.keys())
            ret = test_model(model, users_to_test, drop_flag=False)
            print(ret)


def train_mpt():
    # Write code below
    model = NGCF(data.n_users, data.n_items, norm_adj, args).to('cuda')
    trainer = Trainer(model, args)


if __name__ == '__main__':
    # args = parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using " + str(args.device) + " for computations")

    # data.read_dataset()
    #
    # data.create_adj_mat()
    norm_adj = data.get_adj_mat()

    args.node_dropout = eval(args.node_dropout)
    args.message_dropout = eval(args.message_dropout)

    train_mpt()