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
