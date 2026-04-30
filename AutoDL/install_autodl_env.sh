#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/autodl-fs/ringsharp"
AUTODL_DIR="${PROJECT_ROOT}/AutoDL"
RINGSHARP_DIR="${PROJECT_ROOT}/RINGSharp"
FAST_GICP_DIR="${PROJECT_ROOT}/fast_gicp"
TORCH_RADON_DIR="${PROJECT_ROOT}/torch-radon"
EXTERNAL_DIR="${PROJECT_ROOT}/external"

cd "${PROJECT_ROOT}"

# 1. 清理你本地 Docker 构建时使用的代理变量
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

# 2. 加载环境变量
source "${AUTODL_DIR}/env_autodl.sh"

# 3. 如果 AutoDL 镜像有 conda，优先使用 conda base 环境，保证 faiss-gpu 和 pip 包装在同一个 Python 里
if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate base
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate base
fi

PYTHON_BIN="$(command -v python)"
echo "[INFO] Using Python: ${PYTHON_BIN}"
python -V

# 4. 替换 Ubuntu 源为阿里云源
if [ -f /etc/apt/sources.list ]; then
    sed -i         -e 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g'         -e 's|http://security.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g'         /etc/apt/sources.list
fi

# 5. 安装系统依赖
apt-get     -o Acquire::Retries=5     -o Acquire::http::Proxy=false     -o Acquire::https::Proxy=false     update

apt-get     -o Acquire::Retries=5     -o Acquire::http::Proxy=false     -o Acquire::https::Proxy=false     install -y --no-install-recommends     sudo apt-utils curl wget vim git     build-essential cmake make pkg-config ninja-build     unzip zip htop tree net-tools iputils-ping     python3-pip python-is-python3 python3-dev python3-distutils     golang-go     libopenblas-dev     libpcl-dev     pybind11-dev     libgl1 libglib2.0-0 libsm6 libxrender1 libxext6     ffmpeg

rm -rf /var/lib/apt/lists/*

# 6. pip 基础配置
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
python -m pip config set global.timeout 120

python -m pip install --no-cache-dir --upgrade "pip<25.1" wheel "setuptools==67.6.0"
python -m pip install --no-cache-dir -U testresources
python -m pip install --no-cache-dir typing-extensions==4.7.1

# 7. 安装 PyTorch 1.11.0 + CUDA 11.3
python -m pip install --no-cache-dir     -f https://mirrors.aliyun.com/pytorch-wheels/cu113/     torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0

# 8. 检查 PyTorch/CUDA
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY

# 9. 安装 OpenMMLab 栈
python -m pip install --no-cache-dir     -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html     mmcv-full==1.7.0

python -m pip install --no-cache-dir     mmdet==2.28.0     mmsegmentation==0.30.0

python -m pip install --no-cache-dir     matplotlib==3.5.3     Shapely==1.8.5.post1     networkx==2.2     scikit-image==0.19.3     wandb

python -m pip install --no-cache-dir     mmdet3d==1.0.0rc4

# 10. 安装你的 requirements
python -m pip install --no-cache-dir -r "${AUTODL_DIR}/requirements_autodl.txt"

# 11. faiss-gpu 单独安装
# 注意：这一步要求当前 python 和 conda base 是同一个环境。
if command -v conda >/dev/null 2>&1; then
    conda install -y -c pytorch -c nvidia faiss-gpu=1.7.2 cudatoolkit=11.3 || {
        echo "[WARN] faiss-gpu conda install failed."
        echo "[WARN] 如果代码必须使用 faiss-gpu，请手动检查 conda/python 是否为同一个环境。"
    }
else
    echo "[WARN] conda not found, skip faiss-gpu."
    echo "[WARN] 建议使用带 Miniconda 的 AutoDL 镜像。"
fi

# 12. 清理本地上传过来的旧编译缓存
echo "[INFO] Cleaning local build artifacts..."

rm -rf "${RINGSHARP_DIR}/build"        "${RINGSHARP_DIR}/dist"        "${RINGSHARP_DIR}"/*.egg-info

rm -rf "${FAST_GICP_DIR}/build"        "${FAST_GICP_DIR}/dist"        "${FAST_GICP_DIR}"/*.egg-info

rm -rf "${TORCH_RADON_DIR}/build"        "${TORCH_RADON_DIR}/dist"        "${TORCH_RADON_DIR}/objs"        "${TORCH_RADON_DIR}/__pycache__"        "${TORCH_RADON_DIR}"/*.egg-info

# 13. 安装 fast_gicp
if [ -f "${FAST_GICP_DIR}/setup.py" ]; then
    echo "[INFO] Installing fast_gicp..."
    cd "${FAST_GICP_DIR}"
    python -m pip install --no-cache-dir -v -e .
else
    echo "[WARN] ${FAST_GICP_DIR}/setup.py not found, skip fast_gicp."
fi

# 14. 安装 torch-radon
if [ -f "${TORCH_RADON_DIR}/setup.py" ]; then
    echo "[INFO] Installing torch-radon..."
    cd "${TORCH_RADON_DIR}"
    python -m pip install --no-cache-dir -v -e .
else
    echo "[WARN] ${TORCH_RADON_DIR}/setup.py not found, skip torch-radon."
fi

# 15. 安装 RINGSharp 主项目
if [ -f "${RINGSHARP_DIR}/setup.py" ]; then
    echo "[INFO] Installing RINGSharp..."
    cd "${RINGSHARP_DIR}"
    python -m pip install --no-cache-dir -v -e .
else
    echo "[WARN] ${RINGSHARP_DIR}/setup.py not found, skip RINGSharp editable install."
fi

# 16. 按原 Docker 环境补充外部依赖：OpenPCDet 和 MinkowskiEngine
mkdir -p "${EXTERNAL_DIR}"
cd "${EXTERNAL_DIR}"

if [ ! -d "${EXTERNAL_DIR}/OpenPCDet" ]; then
    echo "[INFO] Cloning OpenPCDet..."
    git clone https://github.com/open-mmlab/OpenPCDet.git
fi

if [ -d "${EXTERNAL_DIR}/OpenPCDet" ]; then
    echo "[INFO] Installing OpenPCDet..."
    cd "${EXTERNAL_DIR}/OpenPCDet"
    python -m pip install --no-cache-dir -r requirements.txt || true
    python setup.py develop
fi

cd "${EXTERNAL_DIR}"

if [ ! -d "${EXTERNAL_DIR}/MinkowskiEngine" ]; then
    echo "[INFO] Cloning MinkowskiEngine..."
    git clone --recursive https://github.com/NVIDIA/MinkowskiEngine.git
fi

if [ -d "${EXTERNAL_DIR}/MinkowskiEngine" ]; then
    echo "[INFO] Installing MinkowskiEngine..."
    cd "${EXTERNAL_DIR}/MinkowskiEngine"
    mkdir -p MinkowskiEngineBackend
    touch MinkowskiEngineBackend/__init__.py

    python -m pip install --no-cache-dir -U "pip<25.1" wheel "setuptools==67.6.0"
    python -m pip install --no-cache-dir -v -e .
fi

# 17. 写入 ~/.bashrc
grep -q "ringsharp/AutoDL/env_autodl.sh" ~/.bashrc ||     echo "source /root/autodl-fs/ringsharp/AutoDL/env_autodl.sh" >> ~/.bashrc

echo ""
echo "============================================================"
echo "AutoDL RINGSharp environment installation finished."
echo "Project root: ${PROJECT_ROOT}"
echo "请执行：source ~/.bashrc"
echo "然后进入：cd /root/autodl-fs/ringsharp"
echo "============================================================"
