# downlaod flsh-attn
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
FLASH_ATTN_WHL="flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"

pip install "sglang[all]==0.5.2" --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch-memory-saver --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --no-cache-dir "vllm==0.11.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pre-commit ruff tensorboard  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1" -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --no-cache-dir $FLASH_ATTN_WHL
pip install --no-cache-dir flashinfer-python==0.3.1  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-python  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-fixer -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c "from opencv_fixer import AutoFix; AutoFix()"
pip install swanlab
pip install math_verify
"""
solve RuntimeError: Failed to complete async request to release_memory_occupation after 3 attempts
"""
pip install "uvicorn<0.34.0" "starlette<0.42.0"  "accelerate>=1.0.0"

pip install numpy==2.2.6
cd verl
pip install --no-deps -e .

"""
1、ImportError: libnuma.so.1: cannot open shared object file: No such file or directory
apt update
apt install libnuma-dev -y
"""
