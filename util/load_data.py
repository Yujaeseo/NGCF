import scipy.sparse as sp
import numpy as np

class Data():

    def __init__(self, train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file

        self.n_users, self.n_items = 0, 0
        self.n_train_ratings, self.n_test_ratings = 0, 0
        self.R = None
        self.train_items, self.test_set = {}, {}

    def read_dataset(self):
        # With 문 사용 => with 블록을 벗어나면 자동으로 file을 close 해줘야 한다.
        with open(self.train_file) as f:
            for line in f.readlines():
                if len(line) > 0:
                    line = line.strip('\n').split(' ')
                    items = [int(i) for i in line[1:]]
                    uid = int(line[0])
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

        print('# of users     {0: 10}  # of items    {1: 10}'.format(self.n_users, self.n_items))
        print('# of train set {0: 10}  # of test set {1: 10}'.format(self.n_train_ratings, self.n_test_ratings))

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

        print('Train sparse matrix nonzeros {}'.format(self.R.count_nonzero()))
