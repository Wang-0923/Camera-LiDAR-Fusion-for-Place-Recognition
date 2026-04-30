#### 数据预处理

cd glnet/datasets/nclt

python generate_training_tuples.py --dataset_root /path/to/NCLT

python generate_evaluation_sets.py --dataset_root /path/to/NCLT

#### 训练

python3 RINGSharp/tools/train.py \
  --config RINGSharp/glnet/config/config_nclt.txt \
  --model_config RINGSharp/glnet/config/ring_sharp_vl_pr_nclt.txt \
  --exp_name ring_sharp_vl_pr_nclt \
  --dataset_type nclt \
  --dataset_root Data/NCLT

#### 评价

python3 RINGSharp/tools/evaluate_pr.py \
  --dataset_root Data/NCLT \
  --dataset_type nclt \
  --eval_set test_2012-02-04_2012-03-17_20.0_5.0.pickle \
  --model_config RINGSharp/glnet/config/ring_sharp_vl_pr_nclt.txt \
  --weight results/weights/ring_sharp_vl_pr_nclt/<your_weight>.pth \
  --exp_name ring_sharp_vl_pr_nclt