import scipy.sparse as sp
import numpy as np
import random as rd

class Data():

    def __init__(self, train_file, test_file, batch_size):
        self.train_file = train_file
        self.test_file = test_file
        self.batch_size = batch_size

        self.n_users, self.n_items = 0, 0
        self.n_train_ratings, self.n_test_ratings = 0, 0
        self.R = None
        self.train_items, self.test_set = {}, {}

        self.exist_users = []

        self.read_dataset()
        self.create_adj_mat()

    def read_dataset(self):
        # With 문 사용 => with 블록을 벗어나면 자동으로 file을 close 해줘야 한다.
        with open(self.train_file) as f:
            for line in f.readlines():
                if len(line) > 0:
                    line = line.strip('\n').split(' ')
                    items = [int(i) for i in line[1:]]
                    uid = int(line[0])
                    self.exist_users.append(uid)
                    self.n_users = max(self.n_users, uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_train_ratings += len(items)

        with open(self.test_file) as f:
            for line in f.readlines():
                if len(line) > 0:
                    line = line.strip('\n')
                    try:
                        items = [int(i) for i in line.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test_ratings += len(items)

        self.n_items += 1
        self.n_users += 1

        # Sparse matrix format 간 비교 필요
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        # print('# of users     {0: 10}  # of items    {1: 10}'.format(self.n_users, self.n_items))
        # print('# of train set {0: 10}  # of test set {1: 10}'.format(self.n_train_ratings, self.n_test_ratings))

        with open(self.train_file) as f_train:
            with open(self.test_file) as f_test:
                for line in f_train.readlines():
                    if len(line) == 0:
                        break
                    line = line.strip('\n')
                    items = [int(i) for i in line.split(' ')]
                    uid, train_items = items[0], items[1:]

                    for i in train_items:
                        self.R[uid, i] = 1.

                    self.train_items[uid] = train_items

                for line in f_test.readlines():
                    if len(line) == 0:
                        break
                    line = line.strip('\n')
                    try:
                        items = [int(i) for i in line.split(' ')]
                    except Exception:
                        continue

                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items

        # print('Train sparse matrix nonzeros {}'.format(self.R.count_nonzero()))

    def create_adj_mat(self):
        adj_mat = sp.lil_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype = np.float32)
        # print(adj_mat.nonzero())
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def get_normalized_adj_mat (adj):
            row_sum = np.array(adj.sum(1))
            # print(row_sum)
            d_inv = np.power(row_sum, -0.5).flatten()
            # print(d_inv)
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj).dot(d_mat_inv)
            return norm_adj

        norm_adj_mat = get_normalized_adj_mat(adj_mat)
        # ngcf_norm_adj_mat = norm_adj_mat + sp.eye(adj_mat.shape[0])

        return norm_adj_mat.tocsc()

    def get_adj_mat(self):
        try:
            ngcf_norm_adj_mat = sp.load_npz('./Data/' + 's_adj_mat.npz')
            # print('Loaded adjacency-matrix (shape:', ngcf_norm_adj_mat.shape, ')')
        except Exception:
            # print('Creating adjacency-matrix...')
            ngcf_norm_adj_mat = self.create_adj_mat()
            sp.save_npz('./Data/' + 's_adj_mat.npz', ngcf_norm_adj_mat)
        return ngcf_norm_adj_mat

    def sample(self):
        if self.batch_size <= self.n_users:
            # without replacement
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            # replacement
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u (u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id_idx = np.random.randint(low=0, high=n_pos_items, size = 1)[0]
                pos_i_id = pos_items[pos_id_idx]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u (u, num):
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                # 효율화할 수 있지 않을까
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items
