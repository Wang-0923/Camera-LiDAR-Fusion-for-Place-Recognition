#!/usr/bin/env bash

# ===== 项目路径：按你当前目录结构 =====
export PROJECT_ROOT=/root/autodl-fs/ringsharp
export AUTODL_DIR=${PROJECT_ROOT}/AutoDL
export RINGSHARP_DIR=${PROJECT_ROOT}/RINGSharp
export FAST_GICP_DIR=${PROJECT_ROOT}/fast_gicp
export TORCH_RADON_DIR=${PROJECT_ROOT}/torch-radon
export EXTERNAL_DIR=${PROJECT_ROOT}/external

# ===== 基础环境 =====
export DEBIAN_FRONTEND=noninteractive
export TZ=Asia/Shanghai

# ===== CUDA 11.3 =====
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# ===== 编译参数 =====
export MAX_JOBS=12
export SETUPTOOLS_USE_DISTUTILS=stdlib

# 3090/3080: 8.6; A100: 8.0; T4: 7.5
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6+PTX"

# ===== pip 国内镜像 =====
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
export PIP_DEFAULT_TIMEOUT=120

# ===== 不使用你本地 Ubuntu 的代理 =====
# 127.0.0.1:7897 在 AutoDL 里不是你本地电脑，而是 AutoDL 服务器自己
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY
unset all_proxy

# ===== AutoDL 学术资源加速，有则启用 =====
if [ -f /etc/network_turbo ]; then
    source /etc/network_turbo
fi

# ===== Python 路径 =====
export PYTHONPATH="${PROJECT_ROOT}:${RINGSHARP_DIR}:${FAST_GICP_DIR}:${TORCH_RADON_DIR}:${EXTERNAL_DIR}/OpenPCDet:${EXTERNAL_DIR}/MinkowskiEngine:${PYTHONPATH}"
