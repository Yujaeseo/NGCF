import torch
from util.load_data import *
from util.parser import parse_args
from NGCF.NGCF import NGCF
import torch.optim as optim


if __name__ == '__main__':
    args = parse_args()
    train_file = args.data_path + '/' + args.dataset + '/' + args.train_file
    test_file = args.data_path + '/' + args.dataset + '/' + args.test_file

    data = Data(train_file, test_file, args.batch_size)
    data.read_dataset()

    data.create_adj_mat()
    norm_adj = data.get_adj_mat()

    args.node_dropout = eval(args.node_dropout)
    args.message_dropout = eval(args.message_dropout)

    model = NGCF(data.n_users, data.n_items, norm_adj, args)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epoch):
        print(epoch)
        loss = 0
        n_batch = data.n_train_ratings // args.batch_size + 1

        for idx in range(n_batch):
            print(idx)
            users, pos_items, neg_items = data.sample()
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users, pos_items, neg_items, args)

            batch_loss = model.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            print(batch_loss)
        print(loss)
