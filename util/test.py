def test_model(model, data, users_to_test ,args):

    u_batch_size = args.batch_size * 2
    i_batch_size = args.batch_size

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start : end]

        item_batch = range(data.n_items)

        u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch, item_batch, [], args)
        rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()

    user_batch_rating_uid = zip(rate_batch, user_batch)
    # Test one user
