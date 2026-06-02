#### *数据集放置格式

```
~/autodl-tmp/Data
└── NCLT
    ├── 2012-02-04
    │   ├── ground_truth/groundtruth_2012-02-04.csv
    │   ├── lb3
    │   └── velodyne_sync
    ├── 2012-02-18
    │   ├── ground_truth/groundtruth_2012-03-17.csv
    │   ├── lb3
    │   └── velodyne_sync
    ├── cam_params
    └── U2D
```

#### *数据预处理

```
1. 预处理图像
python glnet/datasets/nclt/image_preprocess.py \
  --dataset_root /root/autodl-tmp/Data/NCLT \
  --seqs 2012-02-04 2012-03-17

2. 生成训练/验证 tuples
python glnet/datasets/nclt/generate_training_tuples.py \
  --dataset_root /root/autodl-tmp/Data/NCLT \
  --sequences 2012-02-04 2012-02-18 \
  --bev \
  --lidar_reliability

python glnet/datasets/nclt/generate_evaluation_sets.py \
  --dataset_root /root/autodl-tmp/Data/NCLT \
  --map_sequence 2012-02-04 \
  --query_sequence 2012-02-18 \
  --bev \
  --lidar_reliability
```

#### *训练

```
python3 tools/train.py \
  --config glnet/config/config_nclt.txt \
  --model_config glnet/config/ring_sharp_vl_pr_nclt.txt \
  --exp_name ring_sharp_vl_pr_nclt \
  --dataset_type nclt \
  --dataset_root /root/autodl-tmp/Data/NCLT \
  --weight xxx.pth
```

#### *验证

```
python3 RINGSharp/tools/evaluate_pr.py \
  --dataset_root Data/NCLT \
  --dataset_type nclt \
  --eval_set test_2012-02-04_2012-03-17_20.0_5.0.pickle \
  --model_config RINGSharp/glnet/config/ring_sharp_vl_pr_nclt.txt \
  --weight results/weights/ring_sharp_vl_pr_nclt/<your_weight>.pth \
  --exp_name ring_sharp_vl_pr_nclt
  --revisit_threshold 5.0
```

