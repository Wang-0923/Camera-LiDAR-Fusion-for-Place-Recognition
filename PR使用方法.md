#### 论文框图

![Overview](/home/a/ringsharp/Overview.jpg)

#### 本地配置Docker环境

```
cd /home/a/ringsharp/docker
bash build.sh
bash run.sh
```

#### *租服务器Autodl中配置环境

```
cd /root/autodl-fs/ringsharp
chmod +x AutoDL/env_autodl.sh
chmod +x AutoDL/install_autodl_env.sh
bash AutoDL/install_autodl_env.sh 2>&1 | tee \ AutoDL/install_log_$(date +%Y%m%d_%H%M%S).txt
source ~/.bashrc
```

#### 终端远程连接Autodl

```
ssh -p 27027 root@connect.bjb1.seetacloud.com
```

#### *第一次上传本地文件到Autodl

```
scp -rP 27027 /home/a/ringsharp root@connect.bjb1.seetacloud.com:/root/autodl-fs/ringsharp
```

#### *将本地主仓库/RINGSharp下代码变化同步到Autodl中

```
1. 在本地执行：
rsync -avz \
  -e "ssh -p 27027" \
  --exclude "build/" \
  --exclude "glnet.egg-info/" \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  /home/a/ringsharp/RINGSharp/ \
  root@connect.bjb1.seetacloud.com:/root/autodl-fs/ringsharp/RINGSharp/

2. 在Autodl实例中执行：
cd /root/autodl-fs/ringsharp
source ~/.bashrc
```

#### VScode使用方法

```
1. 本地使用Codex
export HTTP_PROXY="http://127.0.0.1:7897"
export HTTPS_PROXY="http://127.0.0.1:7897"
export NO_PROXY="localhost,127.0.0.1"
code

2. VScode连接AutoDL：Remote-SSH
```

#### *安装依赖

```
cd fast_gicp
python setup.py install --user

cd ../torch-radon
python setup.py install

cd ../RINGSharp/glnet/ops
pip install -v -e .

cd ../..
python setup.py develop
```

#### *数据集下载

```
1. 先将数据集下载到本地

2. 将本地数据集上传到阿里云盘
仓库：https://github.com/tickstep/aliyunpan（需下载）
cd aliyunpan-v0.3.9-linux-amd64/
 ./aliyunpan
login
upload /本例数据路径 /云端路径

3. 将阿里云盘数据导入AutoDL，参考文档：https://www.autodl.com/docs/
```

#### *数据集放置格式

```
~/autodl-tmp/Data
└── NCLT
    ├── 2012-02-04
    │   ├── ground_truth/groundtruth_2012-02-04.csv
    │   ├── lb3
    │   └── velodyne_sync
    ├── 2012-03-17
    │   ├── ground_truth/groundtruth_2012-03-17.csv
    │   ├── lb3
    │   └── velodyne_sync
    ├── cam_params
    └── U2D
```

#### *数据预处理

```
1. 设置环境变量
cd /root/autodl-fs/ringsharp/RINGSharp
export RINGSHARP_DATA_ROOT=/root/autodl-tmp/Data

2. 预处理图像
python glnet/datasets/nclt/image_preprocess.py \
  --dataset_root /root/autodl-tmp/Data/NCLT \
  --seqs 2012-02-04 2012-03-17

3. 生成训练/验证 tuples
python glnet/datasets/nclt/generate_training_tuples.py \
  --dataset_root /root/autodl-tmp/Data/NCLT

python glnet/datasets/nclt/generate_evaluation_sets.py \
  --dataset_root /root/autodl-tmp/Data/NCLT
```

#### *训练

```
python3 tools/train.py \
  --config glnet/config/config_nclt.txt \
  --model_config glnet/config/ring_sharp_vl_pr_nclt.txt \
  --exp_name ring_sharp_vl_pr_nclt \
  --dataset_type nclt \
  --dataset_root Data/NCLT
```

#### 训练中查看loss曲线

```
1. 在 AutoDL 终端里运行：
tensorboard \
  --logdir /root/autodl-fs/ringsharp/RINGSharp/results/tensorboard/ring_sharp_vl_pr_nclt \
  --host 0.0.0.0 \
  --port 6006
  
2. 在本地终端运行：
ssh -p 22009 -L 6006:127.0.0.1:6006 root@connect.bjb2.seetacloud.com

3. 本地浏览器打开：http://127.0.0.1:6006
```

#### *训练结果保存到本地

```
rsync -avP \
  -e "ssh -p 22009" \
  root@connect.bjb2.seetacloud.com:/root/autodl-fs/ringsharp/RINGSharp/results/ \
  /home/a/ringsharp/RINGSharp/results/
```

#### 训练后在本地查看loss曲线

```
1. 本地终端执行：
cd /home/a/ringsharp/RINGSharp
tensorboard --logdir /home/a/ringsharp/RINGSharp/results/tensorboard/ring_sharp_vl_pr_nclt --host 127.0.0.1 --port 6006

2. 浏览器打开：http://127.0.0.1:6006
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
```