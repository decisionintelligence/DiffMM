import numpy as np

from utils.spatial_func import distance, SPoint, rate2gps


def shrink_seq(seq):
    """remove repeated ids"""
    s0 = seq[0]
    new_seq = [s0]
    for s in seq[1:]:
        if s == s0:
            continue
        else:
            new_seq.append(s)
        s0 = s

    return new_seq


def memoize(fn):
    '''
    Return a memoized version of the input function.

    The returned function caches the results of previous calls.
    Useful if a function call is expensive, and the function
    is called repeatedly with the same arguments.
    '''
    cache = dict()

    def wrapped(*v):
        key = tuple(v)  # tuples are hashable, and can be used as dict keys
        if key not in cache:
            cache[key] = fn(*v)
        return cache[key]

    return wrapped


def lcs(xs, ys):
    '''Return the longest subsequence common to xs and ys.

    Example
    >>> lcs("HUMAN", "CHIMPANZEE")
    ['H', 'M', 'A', 'N']
    '''

    @memoize
    def lcs_(i, j):
        if i and j:
            xe, ye = xs[i - 1], ys[j - 1]
            if xe == ye:
                return lcs_(i - 1, j - 1) + [xe]
            else:
                return max(lcs_(i, j - 1), lcs_(i - 1, j), key=len)
        else:
            return []

    return lcs_(len(xs), len(ys))


def cal_id_acc(predict, target):
    """
    Calculate RID accuracy between predicted and targeted RID sequence.
    1. no repeated rid for two consecutive road segments
    2. longest common subsequence
    http://wordaligned.org/articles/longest-common-subsequence
    Args:
    -----
        predict = [seq len]
        target = [seq len]
        predict and target have been removed sos
    Returns:
    -------
        mean matched RID accuracy.
    """
    assert len(predict) == len(target)
    ttl = len(predict)
    cnt = np.sum(np.array(predict) == np.array(target))

    # compute average rid accuracy
    shr_trg_ids = shrink_seq(target)
    shr_pre_ids = shrink_seq(predict)
    correct_id_num = len(lcs(shr_trg_ids, shr_pre_ids))
    ttl_trg_id_num = len(shr_trg_ids)
    ttl_pre_id_num = len(shr_pre_ids)

    rid_acc = cnt / ttl
    rid_recall = correct_id_num / ttl_trg_id_num
    rid_precision = correct_id_num / ttl_pre_id_num
    if rid_precision + rid_recall < 1e-6:
        rid_f1 = 0
    else:
        rid_f1 = 2 * rid_recall * rid_precision / (rid_precision + rid_recall)
    return rid_acc, rid_recall, rid_precision, rid_f1


def cal_rn_dis_loss(predict_gps, predict_id, target_gps, target_id):
    """
    Calculate road network based MAE and RMSE between predicted and targeted GPS sequence.
    Args:
    -----
        sp_solver: shortest path solver
        predict_gps = [seq len, 2]
        predict_id = [seq len]
        target_gps = [seq len, 2]
        target_id = [seq len]

        predict and target have been removed sos
    Returns:
    -------
        MAE in meter.
        RMSE in meter.
    """
    ls_dis = []
    assert len(predict_id) == len(target_id) and len(predict_gps) == len(target_gps)
    trg_len = len(predict_gps)

    for i in range(trg_len):
        ls_dis.append(distance(SPoint(*predict_gps[i]), SPoint(*target_gps[i])))

    ls_dis = np.array(ls_dis)

    mae = ls_dis.mean()
    rmse = np.sqrt((ls_dis ** 2).mean())
    return mae, rmse


def toseq(rn, rids, rates, path, seg_info):
    """
    Convert batched rids and rates to gps sequence.
    Args:
    -----
    rn_dict:
        use for rate2gps()
    rids:
        [trg len, batch size, id one hot dim] in torch
    rates:
        [trg len, batch size] in torch
    Returns:
    --------
    seqs:
        [trg len, batch size, 2] in torch
    """

    seqs = []
    # ttl_length = seg_info.get_path_distance(path)
    for seg, rate in zip(rids, rates):
        if seg != 0:
            # idx = path.index(seg)
            # prev_length = seg_info.get_path_distance(path[:idx])
            # curr_length = seg_info.get_seg_length(seg)
            # r0 = max(min((rate * ttl_length - prev_length) / curr_length, 1 - 1e-6), 0)
            r0 = rate
            pt = rate2gps(rn, seg, r0)
            seqs.append([pt.lat, pt.lng])
        else:
            seqs.append([(rn.zone_range[0] + rn.zone_range[2]) / 2, (rn.zone_range[1] + rn.zone_range[3]) / 2])
    return seqs


def calc_metrics(pred_seg, pred_gps, trg_id, trg_gps):
    rid_acc, rid_recall, rid_precision, rid_f1 = cal_id_acc(pred_seg, trg_id)
    mae, rmse = cal_rn_dis_loss(pred_gps, pred_seg, trg_gps, trg_id)
    return rid_recall, rid_precision, rid_f1, rid_acc, mae, rmse
