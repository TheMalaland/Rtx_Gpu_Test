# RTX GPU Test

This project was tested on **January 14-16th, 2025** to verify if older versions of PyTorch work on an **RTX 3050TI Mobile** with an **AMD Ryzen 7 5800H Mobile** in an **Asus TUF A15 laptop**.

---

## Introduction

I noticed that many people face constant issues when installing the correct NVIDIA drivers (CUDA, NVCC, Toolkit, PyTorch). I personally spent hours setting everything up, so I decided to create this guide to help others.

I highly recommend using **Miniconda** for this setup as it allows isolating previously installed variables and dependencies that might interfere with the program. While I plan to add a branch for setups without Conda, I found that for most projects, using Conda is the easiest and most reliable approach.

This guide has been tested in two scenarios:
1. A system with 2 years of intensive use and multiple Conda versions installed.
2. A clean installation of Windows 10.

---

## Prerequisites

It is highly recommended to use **Miniconda** to avoid issues with larger suites like Anaconda.

### Steps to Set Up the Environment:

1. **Install Miniconda (latest version)**  
   - Download Miniconda from the official website:  
     [Miniconda Installation Guide for Windows](https://www.anaconda.com/docs/getting-started/miniconda/install#windows-installation)  
   - I used the silent installation via PowerShell:  
     [Silent Install Instructions](https://www.anaconda.com/docs/getting-started/miniconda/install#windows-command-prompt)  

   **IMPORTANT:** Use the default installation settings. Do not change anything.

2. **Add Miniconda to PATH**  
   - Open environment variables by pressing `Win + R`, typing `sysdm.cpl`, or searching for "Environment Variables".  
   - In the menu, click the "Environment Variables" button.  
   - Under "System Variables," find `Path` or `PATH`, select it, and click "Edit."  
   - Add the following paths (replace `YOURUSER` with your username):  
     ```
     C:\Users\YOURUSER\miniconda3
     C:\Users\YOURUSER\miniconda3\Scripts
     C:\Users\YOURUSER\miniconda3\Library\bin
     ```
   - Save the changes.

3. **Verify Conda Installation**  
   Open `cmd` or PowerShell and run:  
   ```bash
   conda --version
   ```
   You should see something like:  
   ```
   Conda 25.1.1
   ```

---

## Create a Test Environment

Run the following command to create a Conda environment with Python 3.9:

```bash
conda create -n yolov8GPUtest python=3.9
```

### Verified Configuration:

- ✅ Conda 25.1.1  
- ✅ Python 3.11.7 (also tested with 3.9)  
- ✅ CUDA 12.4.0  
- ✅ cuDNN 8.9.7.29  
- ✅ PyTorch 2.5.1 with GPU support (`pytorch-cuda=12.4`)  
- ✅ Visual Studio 2022 Community (no plugins)

Activate the environment:
```bash
conda activate yolov8GPUtest
```

Install PyTorch with GPU support:
```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

---

## Testing GPU Functionality

1. Navigate to the `Scripts` folder:
   ```bash
   cd Scripts
   ```

2. Run the GPU test script:
   ```bash
   python gpu_Pytorch_Test.py
   ```

You should see output similar to this:
```
PyTorch version: 2.5.1
CUDA available: ✅
CUDA version: 12.4
Number of GPUs: 1
✅ GPU detected.
GPU Name: NVIDIA GeForce RTX 3050 Ti Laptop GPU
Memory Capacity: 4.29 GB
✅ GPU operation completed successfully.
```

---

## YOLOv8 GPU Test

1. Activate the `yolov8GPUtest` environment:
   ```bash
   conda activate yolov8GPUtest
   ```

2. Navigate to the `Scripts` folder:
   ```bash
   cd Scripts
   ```

3. Install YOLOv8:
   ```bash
   pip install ultralytics
   ```

4. Run the YOLOv8 test script:
   ```bash
   python GPU_yolov8_vid.py
   ```

5. Select option `2` to use the webcam. If the webcam window opens and uses your RTX GPU, your GPU is working correctly.

6. Use the `q` key to close the program and `3` to exit. Results will be saved in the `Results` folder.

---

## Additional Tests and Manual Driver Installation

### Manual Driver Installation Links:
- **CUDA 12.6.3:** [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)  
- **cuDNN 9.5.1:** [cuDNN Archive](https://developer.nvidia.com/cudnn-archive)  
- **Python 3.13:**  
  ```bash
  conda create -n gpu-base python=3.13
  ```
- **PyTorch 2.7.0:**  
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
  ```

### Results for PyTorch 2.7.0:
```
PyTorch version: 2.7.0+cu126
CUDA available: ✅  
CUDA version: 12.6
Number of GPUs: 1
✅ GPU detected.
GPU Name: NVIDIA GeForce RTX 3050 Ti Laptop GPU
Memory Capacity: 4.29 GB
✅ GPU operation completed successfully.
```

---

## System Configuration

### First Test (PyTorch 2.5.1):
- **CUDA:** 12.6.3  
- **cuDNN:** 9.5.1.17  
- **Python:** 3.11.7  
- **PyTorch:** 2.5.1  
- **Conda:** 25.1.1  
- **C++:** C++17  

### Second Test (PyTorch 2.7.0):
- **CUDA:** 12.9  
- **cuDNN:** 9.5.1.17  
- **Python:** 3.13  
- **PyTorch:** 2.7.0  
- **Conda:** 25.1.1  
- **C++:** C++17  

---

## GPU Information

### `nvidia-smi` Output (First Test):
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 566.36                 Driver Version: 566.36         CUDA Version: 12.7     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3050 ...  WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   57C    P0             10W /   75W |     662MiB /   4096MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

### `nvcc --version` Output:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Wed_Oct_30_01:18:48_Pacific_Daylight_Time_2024
Cuda compilation tools, release 12.6, V12.6.85
Build cuda_12.6.r12.6/compiler.35059454_0
```


### `nvidia-smi` Output (Second Test):
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 576.02                 Driver Version: 576.02         CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3050 ...  WDDM  |   00000000:01:00.0  On |                  N/A |
| N/A   55C    P8              6W /   75W |     188MiB /   4096MiB |     17%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```




