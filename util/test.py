import multiprocessing
from functools import partial
import heapq
from util.metric import *

cores = multiprocessing.cpu_count() // 2


def ranklist_by_heapq(user_pos_text, test_items, rating, ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    k_max = max(ks)
    k_max_item_score = heapq.nlargest(k_max, item_score, key=item_score.get)

    r = []
    for i in k_max_item_score:
        if i in user_pos_text:
            r.append(1)
        else:
            r.append(0)

    return r


def get_metrics(user_pos_test, r, ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for k in ks:
        precision.append(precision_at_k(r, k))
        recall.append(recall_at_k(r, k, len(user_pos_test)))

    return {'recall': np.array(recall), 'precision': np.array(precision)}


def test_one_user(x, data, ks):
    rating = x[0]
    u = x[1]

    try:
        training_items = data.train_items[u]
    except Exception:
        training_items = []
    user_pos_test = data.test_set[u]

    all_items = set(range(data.n_items))

    test_items = list(all_items - set(training_items))
    r = ranklist_by_heapq(user_pos_test, test_items, rating, ks)

    return get_metrics(user_pos_test, r, ks)


def test_model(model, data, users_to_test, args):
    ks = eval(args.ks)
    result = {'precision': np.zeros(len(ks)), 'recall': np.zeros(len(ks))}

    u_batch_size = args.batch_size * 2
    i_batch_size = args.batch_size

    pool = multiprocessing.Pool(cores)

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start:end]

        item_batch = range(data.n_items)

        u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch, item_batch, [], args)
        rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()

        user_batch_rating_uid = zip(rate_batch.numpy(), user_batch)
        test_one_user_func = partial(test_one_user, data=data, ks=ks)
        batch_result = pool.map(test_one_user_func, user_batch_rating_uid)
        #print(batch_result)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users

    pool.close()
    return result
