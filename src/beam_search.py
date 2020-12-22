
import torch


def count_pai_from_context_and_tgt_ids(context, tgt_ids):
    res = [0] * 37

    # hand
    for pai in context['hand'][0]:
        if pai != 0:
            res[pai - 1] += 1

    # discards
    for discards_i in context['discards'][0]:
        for pai in discards_i:
            if pai != 0:
                res[pai - 1] += 1

    # melds
    for melds_i in context['melds']:
        for pai in melds_i[0][0]:
            if pai != 0:
                res[pai - 1] += 1

    # doras
    for pai in context['doras'][0]:
        if pai != 0:
            res[pai - 1] += 1

    # tgt_ids
    for pai in tgt_ids:
        if pai != 38:
            res[pai - 1] += 1

    return res


def reject_invalid_token(log_p_list, context, tgt_ids):
    invalid_log_p = -1e10
    alpha = 0.1
    beta = 1.0

    # sep token is invalid
    log_p_list[-1] = invalid_log_p

    # pai count is 4 or less
    pai_count = count_pai_from_context_and_tgt_ids(context, tgt_ids)
    # print(f'pai_count:{pai_count}')
    for pai in range(37):
        if pai_count[pai] > 4:
            log_p_list[pai] = invalid_log_p
        else:
            log_p_list[pai] *= alpha * pai_count[pai] + beta


    return log_p_list


def beam_search(n_hands_list, model, context, beam_width=4):
    # print(f'n_hands_list : {n_hands_list}')
    heap = [
        {
            'tgt_ids': [],
            'log_p': 0.0,
            'n_hands_list_idx': 0,
            'n_hands_idx': 0
        }
    ]
    next_heap = []
    is_beam_search_finished = False
    while not is_beam_search_finished:
        for beam_search_node in heap:
            tgt_ids = beam_search_node['tgt_ids']
            # print(f'tgt:{tgt_ids}')
            log_p_list = model.predict(context, tgt_ids)
            # print(f'len:{len(tgt_ids)}')

            log_p_list = reject_invalid_token(log_p_list, context, tgt_ids)

            # print(f'log_p_list : {log_p_list}')
            # top_k_values, top_k_indices = torch.topk(log_p_list, k=beam_width)
            sep = []
            n_hands_list_idx_diff = 0
            next_n_hands_idx = beam_search_node['n_hands_idx'] + 1

            if beam_search_node['n_hands_idx'] + 1 == n_hands_list[beam_search_node['n_hands_list_idx']]:
                sep = [38]
                n_hands_list_idx_diff = 1
                next_n_hands_idx = 0

            for i, log_p in enumerate(log_p_list):
                next_heap.append({
                    'tgt_ids': tgt_ids + [i + 1] + sep,
                    'log_p': beam_search_node['log_p'] + log_p,
                    'n_hands_list_idx': beam_search_node['n_hands_list_idx'] + n_hands_list_idx_diff,
                    'n_hands_idx': next_n_hands_idx
                })

            is_beam_search_finished = beam_search_node['n_hands_idx'] + 1 == n_hands_list[beam_search_node['n_hands_list_idx']]
            is_beam_search_finished &= beam_search_node['n_hands_list_idx'] >= 2

        # print(f'sorted heap : {sorted(next_heap, key=lambda x:-x["log_p"])[:beam_width]}')
        heap = next_heap
        heap = sorted(heap, key=lambda x:-x['log_p'])[:beam_width]
        next_heap = []

    best_node = sorted(heap, key=lambda x:-x['log_p'])[-1]

    # print('##############################################')
    print(f'best node : {best_node["tgt_ids"]}')
    # print('##############################################')
    return best_node
    # print(f'log_p : {best_node["log_p"]}')
    # return {
    #     'tgt_ids': best_node['tgt_ids'],
    #     'log_p': best_node['log_p']
    # }
