import numpy as np
import torch
import torch.nn as nn


class NGCF(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, args):
        super(NGCF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.node_dropout = args.node_dropout[0]
        self.message_drop_out = args.message_dropout

        self.norm_adj = norm_adj

        self.layers = eval(args.layer_size)
        self.n_layers = len(self.layers)
        self.decay = eval(args.regs)[0]

        self.embedding_dict, self.weight_dict = self.init_weight()

        self.sparse_norm_tensor = self._convert_sp_mat_to_sp_tensor(self.norm_adj)

    def init_weight(self):
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user, self.emb_dim))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item, self.emb_dim)))
        })

        weight_dict = nn.ParameterDict()
        layers = [self.emb_dim] + self.layers

        for k in range(len(self.layers)):
            weight_dict.update({'W_gc_{}'.format(k): nn.Parameter(initializer(torch.empty(layers[k], layers[k+1])))})
            weight_dict.update({'b_gc_{}'.format(k): nn.Parameter(initializer(torch.empty(1, layers[k+1])))})
            weight_dict.update({'W_bi_{}'.format(k): nn.Parameter(initializer(torch.empty(layers[k], layers[k+1])))})
            weight_dict.update({'b_bi_{}'.format(k): nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

        return embedding_dict, weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float)
        # Indice가 꼭 int64 타입이여야 하는가
        i = torch.LongTensor(np.mat([coo.row, coo.col]))
        v = torch.FloatTensor(coo.data)
        # COO 말고 다른 포멧을 썼을 때의 성능 파악
        res = torch.sparse.FloatTensor(i, v, coo.shape)
        return res

    # def forward(self, users, pos_items, neg_items, drop_flag=True):

        #나중에 dropout 적용하기
        # A_hat = self.sparse_norm_tensor

        # (users + items) * emb_dim
        # ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        #
        # all_embeddings = [ego_embeddings]
        #
        # for k in range(self.n_layers):
        #     side_embeddings =  torch.sparse.mm(A_hat, ego_embeddings)
        #
        #     sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_{}'.format(k)]) + self.weight_dict['b_bi_{}'.format(k)]
        #     # 논문 수식이랑 다름
        #     bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
        #     bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_{}'.format(k)]) + self.weight_dict['b_bi_{}'.format(k)]



