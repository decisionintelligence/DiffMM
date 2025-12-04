import pandas as pd
import numpy as np
import tqdm
from ast import literal_eval
import geopandas as gpd
import networkx as nx
import os
import csv
import json


rt_dict = {
    "motorway": 7,
    "motorway_link": 7,
    "trunk": 6,
    "trunk_link": 6,
    "primary": 5,
    "primary_link": 5,
    "secondary": 4,
    "secondary_link": 4,
    "tertiary": 3,
    "tertiary_link": 3,
    "unclassified": 2,
    "residential": 1,
    "living_street": 1,
}


def get_road_type(rt_str):
    avg = "secondary"

    if "[" in rt_str:
        rts = literal_eval(rt_str)
        new_rts = []
        for item in rts:
            if item in rt_dict:
                new_rts.append(item)
        if len(new_rts) == 0:
            new_rts.append(avg)
        codes = []
        for item in new_rts:
            codes.append(rt_dict[item])
        ans_code = np.max(codes)
        ans_desc = new_rts[np.argmax(codes)]
    else:
        if rt_str not in rt_dict:
            rt_str = avg
        ans_code = rt_dict[rt_str]
        ans_desc = rt_str
    return ans_desc, int(ans_code)


def save_map_txt(map_dir, output_dir):
    nodes = gpd.read_file(os.path.join(map_dir, "nodes.shp"))
    index = [i for i in range(nodes.shape[0])]
    nodes["fid"] = np.array(index, dtype=int)
    data = []
    nid_dict = {}
    for i in tqdm.tqdm(range(nodes.shape[0]), desc="node num"):
        tmp = nodes.iloc[i]
        osmid = int(tmp['osmid'])
        fid = int(tmp['fid'])
        x = float(tmp['x'])
        y = float(tmp['y'])

        nid_dict[osmid] = fid
        data.append([fid, y, x])
    with open(os.path.join(output_dir, "nodeOSM.txt"), 'w', newline='') as fp:
        fields_output_file = csv.writer(fp, delimiter='\t')
        fields_output_file.writerows(data)

    edges = gpd.read_file(os.path.join(map_dir, "edges.shp"))
    zone = [180, -180, 90, -90]
    rn_dict = {}
    data = []
    wayType = []
    for i in tqdm.tqdm(range(edges.shape[0]), desc='edge num'):
        tmp = edges.iloc[i]
        eid = int(tmp['fid'])
        u = int(tmp['u'])
        v = int(tmp['v'])
        points = tmp['geometry'].coords
        zone[0] = min(zone[0], np.min(points.xy[0]))
        zone[1] = max(zone[1], np.max(points.xy[0]))
        zone[2] = min(zone[2], np.min(points.xy[1]))
        zone[3] = max(zone[3], np.max(points.xy[1]))

        desc, code = get_road_type(tmp['highway'])
        wayType.append([eid, desc, code])

        row = [eid, nid_dict[u], nid_dict[v]]
        row.append(len(points))
        pts = []
        for lon, lat in points:
            row += [float(lat), float(lon)]
            pts.append([float(lat), float(lon)])
        data.append(row)

        tmp_dict = {"coords": pts, "length": float(tmp['length']), "level": code}
        rn_dict[eid] = tmp_dict
    print(zone)
    with open(os.path.join(output_dir, "rn_dict.json"), 'w') as fp:
        json.dump(rn_dict, fp)
    with open(os.path.join(output_dir, "edgeOSM.txt"), 'w', newline='') as fp:
        fields_output_file = csv.writer(fp, delimiter='\t')
        fields_output_file.writerows(data)
    with open(os.path.join(output_dir, "wayTypeOSM.txt"), 'w', newline='') as fp:
        fields_output_file = csv.writer(fp, delimiter='\t')
        fields_output_file.writerows(wayType)


if __name__ == '__main__':
    data_root = os.path.join(os.getcwd(), '..', 'data')
    city = 'porto'

    # map_dir, e.g. data/porto/map
    map_dir = os.path.join(data_root, city, 'map')
    map_dir_output = os.path.join(data_root, city, 'roadnetwork')
    if not os.path.exists(map_dir_output):
        os.makedirs(map_dir_output)
    save_map_txt(map_dir, map_dir_output)
