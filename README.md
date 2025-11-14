# DiffMM: Efficient Method for Accurate Noisy and Sparse Trajectory Map Matching via One Step Diffusion

## Requirements

- python==3.11
- torch==2.4.0
- rtree==1.0.1
- geopandas==0.14.4
- networkx==3.3
- einops==0.8.0
- tqdm

Datasets are in `./data`.

We provide toy datasets of Porto and Beijing both have 16,000 trajectories in training set and 10,000 trajectories in valid set and testing set, respectively. The corresponding road networks are also provided.

## Data Format

```
timestamp_1, latitude_1, longitude_1, ground_truth_matched_segment_1
...
timestamp_L, latitude_L, longitude_L, ground_truth_matched_segment_L
-{count}

...
```

Use `./preprocess/reformatrn.py` to generate road network from OSM (Open Street Map) shp files (sepecifically, `nodes.shp` and `edges.shp`). Set the correct city name and data path. (This process is not needed for the provided datasets)

We have provided processed road network of Beijing and Porto in `./data/${city}/roadnetwork`.

## Train and Inference with DiffMM

Example of train and test in Porto dataset with sample ratio $r=0.1$:
```bash
python main.py --city porto --keep_ratio 0.1 --epochs 30 --batch_size 512 --gpu_id 0 --train_flag --test_flag
```

Data preprocess will be performed when first run on a new dataset, this might take a while.
