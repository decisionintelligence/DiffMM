import torch
import torch.nn as nn
from .layers import PointEncoder, Attention, Norm


class TrajEncoder(nn.Module):
    def __init__(self, parameters, device):
        super().__init__()
        self.id_size = parameters.id_size
        self.hid_dim = parameters.hid_dim
        # self.grid_num = parameters.grid_num
        self.id_emb_dim = parameters.hid_dim

        self.emb_id = nn.Parameter(torch.rand(self.id_size, self.id_emb_dim))

        self.device = device

        road_emb_input_dim = self.id_emb_dim + 9
        self.road_emb = nn.Sequential(
            nn.Linear(road_emb_input_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            Norm(self.hid_dim)
        )
        self.point_encoder = PointEncoder(parameters)
        self.attn = Attention(self.hid_dim)

        self.output = nn.Linear(2 * parameters.hid_dim, parameters.hid_dim)

    def merge(self, sequences):
        lengths = [len(seq) for seq in sequences]
        dim = sequences[0].size(1)  # get dim for each sequence
        padded_seqs = torch.zeros(len(sequences), max(lengths), dim)

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    def forward(self, src, src_len, src_segs, segs_feat, segs_mask):
        bs = src.size(0)

        src_id_emb = self.emb_id[src_segs]
        src_road_emb = torch.cat((src_id_emb, segs_feat), dim=-1)
        road_emb = self.road_emb(src_road_emb)

        point_encoder_output = self.point_encoder(src, src_len)
        _, attention = self.attn(point_encoder_output, road_emb, road_emb, segs_mask)

        outputs = torch.cat((point_encoder_output, attention), dim=-1)

        return outputs


class ModelAll(nn.Module):
    def __init__(self, encoder, diffusion):
        super(ModelAll, self).__init__()
        self.encoder = encoder
        self.diffusion = diffusion


class ModelAllShortCut(nn.Module):
    def __init__(self, encoder, shortcut):
        super(ModelAllShortCut, self).__init__()
        self.encoder = encoder
        self.shortcut = shortcut


def init_weights(self):
    """
    Here we reproduce Keras default initialization weights for consistency with Keras version
    Reference: https://github.com/vonfeng/DeepMove/blob/master/codes/model.py
    """
    ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
    hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
    b = (param.data for name, param in self.named_parameters() if 'bias' in name)

    for t in ih:
        nn.init.xavier_uniform_(t)
    for t in hh:
        nn.init.orthogonal_(t)
    for t in b:
        nn.init.constant_(t, 0)
