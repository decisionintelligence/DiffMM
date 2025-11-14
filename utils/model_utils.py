import math
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from utils.spatial_func import rate2gps


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def exp_prob(beta, x):
    """
    error distance weight.
    """
    return math.exp(-pow(x, 2) / pow(beta, 2))


def gps2grid(pt, mbr, grid_size):
    """
    mbr:
        MBR class.
    grid size:
        int. in meter
    """
    LAT_PER_METER = 8.993203677616966e-06
    LNG_PER_METER = 1.1700193970443768e-05
    lat_unit = LAT_PER_METER * grid_size
    lng_unit = LNG_PER_METER * grid_size

    lat = pt.lat
    lng = pt.lng
    locgrid_x = int((lat - mbr.min_lat) / lat_unit) + 1
    locgrid_y = int((lng - mbr.min_lng) / lng_unit) + 1

    return locgrid_x, locgrid_y


def get_normalized_t(first_pt, current_pt, time_interval):
    """
    calculate normalized t from first and current pt
    return time index (normalized time)
    """
    t = int(1 + ((current_pt.time - first_pt.time).seconds / time_interval))
    return t


def get_rn_grid(mbr, rn, grid_size):
    rn_grid = []
    rn_grid.append(torch.tensor([[0, 0]]))
    for i in tqdm(range(1, rn.valid_edge_cnt_one)):
        rid = rn.valid_to_origin_one[i]
        cur_grid = []
        for rate in range(1000):
            r = rate / 1000
            gps = rate2gps(rn, rid, r)
            grid_x, grid_y = gps2grid(gps, mbr, grid_size)
            if len(cur_grid) == 0 or [grid_x, grid_y] != cur_grid[-1]:
                cur_grid.append([grid_x, grid_y])
        rn_grid.append(torch.tensor(cur_grid))
    return rn_grid
