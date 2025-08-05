# Jetson Nano PyTorch Installation (JetPack 6.2 / CUDA 12.6)

## Overview
This document summarizes a working method to install PyTorch with GPU support on the original 4GB Jetson Nano running JetPack 6.2 (which includes CUDA 12.6), including necessary prerequisites, installing cuSPARSELt, installing the NVIDIA-provided PyTorch wheel, verifying the install, and handling the optional (and heavy) torchvision build. It also covers troubleshooting, cleanup, and fallback options for when torchvision builds fail or crash.

## Requirements
- Jetson Nano (original 4GB) flashed with JetPack 6.2 (includes CUDA 12.6, cuDNN, TensorRT, etc.)
- Python 3.10 (default on JetPack 6.2)
- `pip3` available
- Internet connection to download NVIDIA wheels and source if needed

## 1. System Preparation
Update packages and ensure basic tooling exists:

```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y python3-pip python3-virtualenv build-essential cmake     libjpeg-dev zlib1g-dev libopenblas-dev libpython3-dev libavcodec-dev     libavformat-dev libswscale-dev
python3 -m pip install --upgrade pip
```

(Optional) Create and activate a clean virtual environment:
```bash
python3 -m virtualenv ~/jetson-pytorch-env
source ~/jetson-pytorch-env/bin/activate
```

## 2. Install cuSPARSELt (Required for recent PyTorch on Jetson)
JetPack 6.2 does not bundle cuSPARSELt by default. Install it manually:

```bash
export CUSPARSELT_VERSION="0.7.1.0"  # adjust if newer recommended version exists
wget https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-${CUSPARSELT_VERSION}-archive.tar.xz
tar -xf libcusparse_lt-linux-aarch64-${CUSPARSELT_VERSION}-archive.tar.xz
sudo cp -a libcusparse_lt-linux-aarch64-${CUSPARSELT_VERSION}-archive/include/* /usr/local/cuda/include/
sudo cp -a libcusparse_lt-linux-aarch64-${CUSPARSELT_VERSION}-archive/lib/* /usr/local/cuda/lib64/
sudo ldconfig
```

Ensure `/usr/local/cuda` points to CUDA 12.6 (the default on JetPack 6.2).

## 3. Install PyTorch (NVIDIA-prebuilt wheel)
There are two different options for installing pytorch. One is via https://pypi.jetson-ai-lab.io/jp6/cu126, which often has outages. To
install using this method (if working):
```bash
wget https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl#sha256=62a1beee9f2f147076a974d2942c90060c12771c94740830327cae705b2595fc

pip3 install torch-2.8.0-cp310-cp310-linux_aarch64.whl
```


If the above method is broken, use NVIDIA’s official Jetson wheel directly:

Example for PyTorch 2.5.0 (adjust if newer compatible wheel is available):
```bash
pip3 install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
```

If using a virtualenv, ensure `pip3` refers to the venv’s pip. Replace the URL if you target a different NVIDIA build (e.g., newer `v62` directory when available).

## 4. Verification
Run the following in Python to validate GPU-enabled PyTorch:

```python
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("cuDNN version:", torch.backends.cudnn.version())
# Simple GPU tensor
x = torch.cuda.FloatTensor(1).zero_()
print("Tensor on GPU:", x)
```

If `cuda.is_available()` is True and no import errors appear, PyTorch is correctly installed and using the Jetson’s GPU.

## 5. Optional: torchvision Installation (from source)
Again, two different options depending on whether or not https://pypi.jetson-ai-lab.io/jp6/cu126 is up.

Option 1:
```bash
wget https://pypi.jetson-ai-lab.io/jp6/cu126/+f/907/c4c1933789645/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl#sha256=907c4c1933789645ebb20dd9181d40f8647978e6bd30086ae7b01febb937d2d1

pip3 install torchvision-0.23.0-cp310-cp310-linux_aarch64.whl
```

Option 2:

### A. Install the prebuilt torchvision wheel
pip3 install --no-cache https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl
- Limiting parallel jobs via `MAKEFLAGS="-j1"` reduces memory spikes that cause crashes.

### B. Verify torchvision
```python
import torchvision
print("torchvision version:", torchvision.__version__)
```


## 6. Troubleshooting
- **cuDNN mismatch error** (e.g., compiled against a different cuDNN than installed):  
  - Option A: Upgrade/downgrade system cuDNN to the expected version via NVIDIA’s cuDNN local installers for Ubuntu 22.04 on Jetson.  
  - Option B: Ensure PyTorch can see its bundled libraries by exporting:  
    ```bash
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python3 -c 'import torch, os; print(os.path.dirname(torch.__file__))')/lib
    ```

- **NumPy compatibility warnings:**  
  ```bash
  pip3 install "numpy<2"
  ```

- **Torch reports `cuda.is_available()` is False**:  
  - Confirm CUDA is present: `nvcc --version`  
  - Ensure the installed wheel was built for the correct CUDA (JetPack 6.2 uses CUDA 12.6)  
  - Check environment; if inside a venv, it must not shadow system CUDA paths inadvertently.

- **`torchvision` build crashes / OOM**:  
  - Ensure swap is active (`swapon --show`).  
  - Limit concurrency (`MAKEFLAGS="-j1"`).  
  - Close other memory-hungry processes.  
  - Retry; builds may succeed after adjusting swap and concurrency.

## 7. Cleanup / Uninstall
To remove PyTorch or torchvision if needed:
```bash
pip3 uninstall torch torchvision
```

If installed with `--user` and broken state persists, manually delete cached build folders in `~/.cache/pip` and user site packages.

## 8. Alternatives
- **Building PyTorch from source:** Very time-consuming on Nano (requires large swap, hours of compile). Only recommended if you cannot find a compatible prebuilt wheel. You must set appropriate flags (e.g., `TORCH_CUDA_ARCH_LIST="5.3"` for Nano’s Maxwell GPU).  
- **Containerized setup:** Use NVIDIA-provided PyTorch containers (NGC / L4T) which bundle working PyTorch + CUDA. This bypasses local Python packaging pain, at the cost of the overhead of container management.  
- **Skip torchvision entirely** if you only need core tensor computation and implement transforms manually.

## 9. Summary of Key Commands

```bash
# System prep
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-virtualenv build-essential cmake libjpeg-dev zlib1g-dev libopenblas-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev

# Optional venv
python3 -m virtualenv ~/jetson-pytorch-env
source ~/jetson-pytorch-env/bin/activate

# cuSPARSELt install
wget https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-0.7.1.0-archive.tar.xz
tar -xf libcusparse_lt-linux-aarch64-0.7.1.0-archive.tar.xz
sudo cp -a libcusparse_lt-linux-aarch64-0.7.1.0-archive/include/* /usr/local/cuda/include/
sudo cp -a libcusparse_lt-linux-aarch64-0.7.1.0-archive/lib/* /usr/local/cuda/lib64/
sudo ldconfig

# PyTorch wheel install (example)
pip3 install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# Increase swap (for heavy builds)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Limit build parallelism
export MAKEFLAGS="-j1"

# Build torchvision (matching version)
git clone --branch release/0.20 https://github.com/pytorch/vision.git torchvision
cd torchvision
export BUILD_VERSION=0.20.0
python3 setup.py install --user --verbose --no-deps # remove user flag if installing to a virtual environment
```

## 10. Installing ONNXRuntime-GPU

To enable GPU acceleration in ONNX Runtime on Jetson Nano, you must install the **onnxruntime-gpu** wheel from the official Jetson AI Lab index. Make sure the Jetson AI Lab PyPI index is reachable and working before attempting installation.

1. **Ensure the Jetson AI Lab index is accessible:**

   ```bash
   pip3 install --upgrade pip
   pip3 install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+f/4eb/e6a8902dc7708/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl#sha256=4ebe6a8902dc7708434b2e1541b3fe629ebf434e16ab5537d1d6a622b42c622b onnxruntime-gpu
   ```

   Replace `<OFFICIAL_JETSON_AI_LAB_URL>` with the official URL for JetPack 6.2 / CUDA 12.6 wheels.

2. **Direct wheel download (optional):** If the index is unavailable, download the appropriate wheel manually and install:

   ```bash
   # Download the wheel for JetPack 6.2 / CUDA 12.6
   wget https://pypi.jetson-ai-lab.io/jp6/cu126/+f/4eb/e6a8902dc7708/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl#sha256=4ebe6a8902dc7708434b2e1541b3fe629ebf434e16ab5537d1d6a622b42c622b
   pip3 install onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl
   ```

3. **Verify the installation:**

   ```python
   import onnxruntime as ort
   print(ort.get_device())                    # should print 'GPU'
   sess = ort.InferenceSession('model.onnx')
   print(sess.get_providers())                # should list 'CUDAExecutionProvider'
   ```

---


## Final Notes
This setup avoids reliance on the unavailable `pypi.jetson-ai-lab.io` / `.dev` community endpoints by using NVIDIA's official distributions and manual source build for extensions. If you run into persistent build problems, consider deferring `torchvision` or using a container for development.




---
