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


## 11. Real-time RGB-D anonymization server (RealSense → Flask MJPEG)

This script opens the Intel RealSense, runs the **AVPrivacyMasker** per frame, and publishes an **MJPEG** stream over HTTP. It keeps latency low by:

- **Two threads:** a capture thread fills a lock-free buffer (`deque(maxlen=1)`), and a publisher thread runs detection → mask → anonymize → JPEG.
- **Aligned RGB-D:** depth frames are aligned to color via `rs.align(rs.stream.color)` so mask pixels line up with RGB.
- **Fast anonymization:** one full-frame mosaic + **copy only masked pixels** + bounded random noise; avoids per-pixel transforms on the full image.
- **ONNX Runtime providers:** prefers TensorRT EP (if available), then CUDA, then CPU. Engines are cached to `~/.ort_trt_cache`.
- **Warm-up:** a dummy inference once on startup builds TRT engines so the first live frame won’t stutter.

### Run

  ```bash
  # from your project root
  python3 feed.py --port 5001 --preview-width 640 --jpeg-quality 80

  # Optional: save frames alongside the stream
  python3 feed.py --save
  ```

#### SSH if remote

  ```bash
  ssh -L 5001:localhost:5001 <user>@<jetson-ip>
  # then open http://localhost:5001/ on your browser
  ```

#### Command-line options:

-	**--det-size:** detector input size (e.g., 640). Larger is slower but can improve small-face recall.
-	**--depth-anon:** also add noise to depth where mask==1 (off by default).
-	**--jpeg-quality:** 60–90 is a good range; higher = larger frames.
-	**--preview-width:** resize for the stream only (does not change inference resolution).
-	**--save:** write original/ and anon/ frames under images/session_<timestamp>_det-size<det>.

#### Directory layout (when --save is on)

    images/
      session_YYYY-MM-DD_HH-MM-SS_det-size640/
        original/  frame_<ms>.jpg
        anon/      frame_<ms>.jpg


### Performance notes (typical 640×480 on Jetson Nano/Orin)

-	**Face detection (SCRFD via ORT+TensorRT):** ~16–20 ms
-	**Mask construction (box-guided):** ~1–4 ms
-	**Anonymize (fast mosaic+noise):** ~20 ms
-	**End-to-end:** ~24–28 FPS (depends on load / provider)


### Troubleshooting

-	**No camera / RealSense error:** install librealsense2 and pyrealsense2; check realsense-viewer. Ensure udev rules are set.
-	**CUDA provider missing:** confirm you installed onnxruntime-gpu for JetPack 6.2; ort.get_available_providers() should list CUDAExecutionProvider (and TensorrtExecutionProvider if present).
-	**Port already in use:** pick another --port or kill the other process.
-	**Stream choppy:** lower --jpeg-quality, use --preview-width 640, or reduce --det-size.

**Security:** the Flask app has no auth. Keep it on a trusted network or use SSH tunneling / a reverse proxy with auth if you must expose it.

---

## 12. How the script works (quick walkthrough)

- **CLI:** parses flags for save paths, detector size, stream quality, etc.
- **Masker:** `AVPrivacyMasker(device="cuda", det_size=(N,N))` configures SCRFD and sets ORT providers (TensorRT → CUDA → CPU).
- **RealSense:** starts a color (BGR8) and depth (Z16) pipeline at 640×480 @ 30 FPS; aligns depth to color.
- **Warm-up:** a single `detect_faces()` call on a dummy frame triggers TRT engine build once.
- **Capture thread:** grabs synchronized color+depth frames, writes into a one-slot queue (`deque(maxlen=1)`), always keeping the most recent frame.
- **Publisher thread:**
  1) `detect_faces(color)` → boxes  
  2) `build_mask_numpy(depth, boxes, kernel, _calc_depth_profile)` → uint8 mask  
  3) `fast_pixelate(color, mask)` → anonymized RGB  
  4) JPEG encode and publish for **/view**
- **Flask:** `/view` yields an MJPEG boundary stream with the newest JPEG in a simple loop.











---
