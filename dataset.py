import os.path
import pickle

import torch
import numpy as np
import tqdm
import random
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from utils.parse_traj import ParseMMTraj
from utils.model_utils import get_normalized_t, gps2grid


class MMDataset(Dataset):
    def __init__(self, rn, traj_dir, mbr, args, mode, device):
        self.rn = rn
        self.mbr = mbr
        self.args = args
        self.seg_size = rn.valid_edge_cnt
        self.device = device

        self.grid_size = args.grid_size
        self.time_span = args.time_span
        self.keep_ratio = args.keep_ratio

        # self.src_trajs = None
        self.src_norm_gps_seq, self.src_segs_id, self.src_segs_feat = [], [], []
        self.trg_rid = []

        self.mode = mode

        self.get_data(traj_dir)

    def __len__(self):
        # return len(self.src_trajs)
        return len(self.src_norm_gps_seq)

    def __getitem__(self, idx):
        # src_segs_id begins from 1, trg_rid begins from 0
        trg_rid = self.trg_rid[idx].clone().detach().to(self.device)
        trg_onehot = torch.zeros((trg_rid.size(0), self.seg_size), device=self.device)
        for i, rid in enumerate(trg_rid):
            trg_onehot[i, rid] = 1

        return (self.src_norm_gps_seq[idx].clone().detach().to(self.device), self.trg_rid[idx].clone().detach().to(self.device), trg_onehot.clone().detach().to(self.device),
                self.src_segs_id[idx].copy(), self.src_segs_feat[idx].copy())
        # src_traj = self.src_trajs[idx]
        #
        # length = len(src_traj.pt_list)
        #
        # keep_index = [0] + sorted(random.sample(range(1, length - 1), int((length - 2) * self.keep_ratio))) + [
        #     length - 1]
        #
        # src_list = np.array(src_traj.pt_list, dtype=object)
        # src_list = src_list[keep_index].tolist()
        #
        # src_gps_seq, src_norm_gps_seq, trg_rid = self.get_src_seq(src_list)
        #
        # src_segs = self.rn.get_src_segs(src_gps_seq, self.args.search_dist, self.args.beta)
        # segs_id, segs_feat = self.get_segs_feats(src_segs)
        # src_norm_gps_seq = torch.tensor(src_norm_gps_seq)
        #
        # trg_onehot = torch.zeros((length, self.seg_size))
        # for i, rid in enumerate(trg_rid):
        #     trg_onehot[i, rid] = 1
        # trg_rid = torch.tensor(trg_rid)
        # return src_norm_gps_seq, trg_rid, trg_onehot, segs_id, segs_feat

    def get_data(self, traj_dir):
        parser = ParseMMTraj(self.rn)

        if self.mode == 'train':
            src_file = os.path.join(traj_dir, 'traj_train.txt')
        elif self.mode == 'valid':
            src_file = os.path.join(traj_dir, 'traj_valid.txt')
        elif self.mode == 'test':
            src_file = os.path.join(traj_dir, 'traj_test.txt')
        else:
            raise NotImplementedError

        pkl_path = os.path.join(traj_dir, self.mode + '_data_' + str(self.args.search_dist) + '_' + str(self.keep_ratio) + '.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as fp:
                self.src_norm_gps_seq, self.trg_rid, self.src_segs_id, self.src_segs_feat = pickle.load(fp)

            # if self.mode == 'train':
            #     self.src_norm_gps_seq, self.trg_rid, self.src_segs_id, self.src_segs_feat = self.src_norm_gps_seq[:128000], self.trg_rid[:128000], self.src_segs_id[:128000], self.src_segs_feat[:128000]

            if self.mode == 'valid':
                self.src_norm_gps_seq, self.trg_rid, self.src_segs_id, self.src_segs_feat = self.src_norm_gps_seq[:10000], self.trg_rid[:10000], self.src_segs_id[:10000], self.src_segs_feat[:10000]

            # if self.mode == 'test':
            #     self.src_norm_gps_seq, self.trg_rid, self.src_segs_id, self.src_segs_feat = self.src_norm_gps_seq[:1], self.trg_rid[:1], self.src_segs_id[:1], self.src_segs_feat[:1]
            #     pickle.dump(self.trg_rid[:1], open('rid.pkl', 'wb'))
            #     pickle.dump(self.src_norm_gps_seq[:1], open('gps.pkl', 'wb'))
        else:
            src_trajs = parser.parse(src_file, is_target=True)

            for src_traj in tqdm.tqdm(src_trajs, desc='traj num'):
                length = len(src_traj.pt_list)
                keep_index = [0] + sorted(random.sample(range(1, length - 1), int((length - 2) * self.keep_ratio))) + [
                    length - 1]

                src_list = np.array(src_traj.pt_list, dtype=object)
                src_list = src_list[keep_index].tolist()

                src_gps_seq, src_norm_gps_seq, trg_rid = self.get_src_seq(src_list)

                src_segs = self.rn.get_src_segs(src_gps_seq, self.args.search_dist, self.args.beta)
                segs_id, segs_feat = self.get_segs_feats(src_segs)
                src_norm_gps_seq = torch.tensor(src_norm_gps_seq)

                # trg_onehot = torch.zeros((length, self.seg_size))
                # for i, rid in enumerate(trg_rid):
                #     trg_onehot[i, rid] = 1
                trg_rid = torch.tensor(trg_rid)

                self.src_norm_gps_seq.append(src_norm_gps_seq)
                self.trg_rid.append(trg_rid)
                # self.trg_onehot.append(trg_onehot)
                self.src_segs_id.append(segs_id)
                self.src_segs_feat.append(segs_feat)

            with open(pkl_path, 'wb') as fp:
                pickle.dump((self.src_norm_gps_seq, self.trg_rid, self.src_segs_id, self.src_segs_feat), fp)
                
            if self.mode == 'valid':
                self.src_norm_gps_seq, self.trg_rid, self.src_segs_id, self.src_segs_feat = self.src_norm_gps_seq[:10000], self.trg_rid[:10000], self.src_segs_id[:10000], self.src_segs_feat[:10000]

    def get_src_seq(self, ds_pt_list):
        ls_gps_seq = []
        ls_norm_gps_seq = []
        mm_eids = []
        mm_onehot_eids = []
        first_pt = ds_pt_list[0]
        last_pt = ds_pt_list[-1]
        time_interval = self.time_span

        for ds_pt in ds_pt_list:
            ls_gps_seq.append([ds_pt.lat, ds_pt.lng])

            normed_lat = (ds_pt.lat - self.rn.minLat) / (self.rn.maxLat - self.rn.minLat)
            normed_lng = (ds_pt.lng - self.rn.minLon) / (self.rn.maxLon - self.rn.minLon)
            t = get_normalized_t(first_pt, ds_pt, time_interval)
            ls_norm_gps_seq.append([normed_lat, normed_lng, t])

            mm_eids.append(ds_pt.data['candi_pt'].eid)

        return ls_gps_seq, ls_norm_gps_seq, mm_eids

    def get_segs_feats(self, ls_seg):
        seg_id = []
        seg_feat = []

        for segs in ls_seg:
            tmp_id = []
            tmp_feat = []
            for seg in segs:
                tmp_id.append(seg.eid + 1)
                tmp_feat.append([seg.err_weight, seg.cosv, seg.cosv_pre, seg.cosf, seg.cosl, seg.cos1, seg.cos2, seg.cos3, seg.cosp])

            seg_id.append(tmp_id)
            seg_feat.append(tmp_feat)

        return seg_id, seg_feat


def collate_fn(data, device):
    norm_gps_seq, trg_rid, trg_onehot, segs_id, segs_feat = zip(*data)
    lengths = [len(seq) for seq in norm_gps_seq]
    segs_len = [len(seg) for seq in segs_id for seg in seq]
    max_len = max(segs_len)
    segs_id = list(segs_id)
    segs_feat = list(segs_feat)
    segs_mask = []
    for i, segs_seq in enumerate(segs_id):
        tmp_mask = []
        for j, segs in enumerate(segs_id[i]):
            tmp_mask.append([1] * len(segs) + [0] * (max_len - len(segs)))
            segs_id[i][j] = torch.cat((torch.tensor(segs_id[i][j]), torch.zeros(max_len - len(segs))), dim=-1).tolist()
        segs_id[i] = torch.tensor(segs_id[i], device=device)
        segs_mask.append(torch.tensor(tmp_mask, device=device))

    feat_dim = len(segs_feat[0][0][0])
    for i, feats_seq in enumerate(segs_feat):
        for j, feats in enumerate(feats_seq):
            segs_feat[i][j] = torch.cat((torch.tensor(segs_feat[i][j]), torch.zeros((max_len-len(feats), feat_dim))), dim=-2).tolist()
        segs_feat[i] = torch.tensor(segs_feat[i], device=device)

    norm_gps_seq = pad_sequence(norm_gps_seq, batch_first=True, padding_value=0)
    trg_rid = pad_sequence(trg_rid, batch_first=True, padding_value=0).int()
    trg_onehot = pad_sequence(trg_onehot, batch_first=True, padding_value=0)
    segs_id = pad_sequence(segs_id, batch_first=True, padding_value=0).int()
    segs_feat = pad_sequence(segs_feat, batch_first=True, padding_value=0)
    segs_mask = pad_sequence(segs_mask, batch_first=True, padding_value=0)
    lengths = torch.tensor(lengths, dtype=torch.int)

    return lengths, norm_gps_seq, trg_rid, trg_onehot, segs_id, segs_feat, segs_mask
