import torch
from util.load_data import *
from util.parser import parse_args
from NGCF.NGCF import NGCF
import torch.optim as optim
from time import time
from util.test import *

if __name__ == '__main__':
    args = parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using " + str(args.device) + " for computations")

    # Create adj matrix
    norm_adj = data.get_adj_mat()

    args.node_dropout = eval(args.node_dropout)
    args.message_dropout = eval(args.message_dropout)

    model = NGCF(data.n_users, data.n_items, norm_adj, args).to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0

    if args.resume != 0:
        checkpoint_info = torch.load(f'./Checkpoint_mod/{args.checkpoint_prefix}_epoch {args.resume}.pth')
        model.load_state_dict(checkpoint_info['model_state_dict'])
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        optimizer.load_state_dict(checkpoint_info['optimizer_state_dict'])
        start_epoch = checkpoint_info['epoch'] + 1

    for epoch in range(start_epoch, args.epoch):
        t0_start = time()
        loss = 0
        n_batch = data.n_train_ratings // args.batch_size + 1

        for idx in range(n_batch):
            users, pos_items, neg_items = data.sample()
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users, pos_items, neg_items,
                                                                           drop_flag=args.node_dropout)

            batch_loss = model.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss

        t0_end = time()

        print('epoch {} : loss {} , time {}s'.format(epoch + 1, loss.item(), t0_end - t0_start))

        if (epoch + 1) % 50 == 0:
            users_to_test = list(data.test_set.keys())
            ret = test_model(model, users_to_test, drop_flag=False)
            print(ret)

        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'model_state_dict': model.state_dict()
            }, f'./Checkpoint_mod/{args.checkpoint_prefix}_epoch {epoch + 1}.pth')
            print('Save checkpoint')