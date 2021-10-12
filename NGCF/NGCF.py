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