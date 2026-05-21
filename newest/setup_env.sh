#!/bin/bash

# 远程 A800 服务器环境搭建脚本
# 在远程服务器上运行此脚本

set -e

echo "=========================================="
echo "MoE-Mixed-Prec 环境搭建"
echo "=========================================="

# 1. 安装 Miniconda（如果没有）
if ! command -v conda &> /dev/null; then
    echo "[1/5] 安装 Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    rm /tmp/miniconda.sh
    echo "Miniconda 安装完成，请重新打开终端后再次运行此脚本"
    exit 0
else
    echo "[1/5] Miniconda 已安装"
    eval "$(conda shell.bash hook)"
fi

# 2. 创建 conda 环境
echo "[2/5] 创建 conda 环境 moe_mixed_prec (Python 3.10)..."
if conda env list | grep -q "moe_mixed_prec"; then
    echo "  环境已存在，跳过创建"
else
    conda create -n moe_mixed_prec python=3.10 -y
fi
conda activate moe_mixed_prec

# 3. 安装 PyTorch (CUDA 12.x)
echo "[3/5] 安装 PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. 安装项目依赖
echo "[4/5] 安装项目依赖..."
PROJECT_ROOT="$HOME/MoE-Mixed-Prec-main"

# 核心依赖
pip install transformers==4.38.2 \
    accelerate \
    safetensors \
    sentencepiece \
    tokenizers \
    datasets \
    huggingface-hub \
    einops \
    timm

# 量化相关
pip install auto_round

# 数据处理（仅 numpy，其余由 VLMEvalKit 按需拉取）
pip install numpy

# 5. 安装 VLMEvalKit (editable mode，会自动安装其依赖)
echo "[5/5] 安装 VLMEvalKit..."
cd "$PROJECT_ROOT/04_evaluation/VLMEvalKit"
pip install -e .

# 锁定 transformers 版本（VLMEvalKit 可能安装了不兼容的版本）
pip install transformers==4.38.2

# 验证
echo ""
echo "=========================================="
echo "环境验证"
echo "=========================================="
python -c "
import torch
print(f'Python: {__import__(\"sys\").version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
import transformers; print(f'Transformers: {transformers.__version__}')
import auto_round; print(f'AutoRound: {auto_round.__version__}')
print('所有依赖安装成功！')
"

echo ""
echo "环境搭建完成！"
echo "下一步: 运行 bash run_owen_pipeline_remote.sh"
