# RTX GPU Test

This project was tested on **January 14-16th, 2025** to verify if older versions of PyTorch work on an **RTX 3050TI Mobile** with an **AMD Ryzen 7 5800H Mobile** in an **Asus TUF A15 laptop**.

---

## Prerequisites

It is highly recommended to use **Miniconda** to avoid issues with larger suites like Anaconda.

### Steps to set up the environment:

1. **Install Miniconda (latest version)**  
   You can find Miniconda here:  
   [Miniconda Installation Guide for Windows](https://www.anaconda.com/docs/getting-started/miniconda/install#windows-installation)  

   I used the silent installation via PowerShell:  
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

3. **Verify Conda installation**  
   Open `cmd` or PowerShell and run:  
   ```bash
   conda --version
   ```
   You should see something like:  
   ```
   Conda 25.1.1
   ```

---

## Create a test environment

Run the following command to create a Conda environment with Python 3.9:

```bash
conda create -n yolov8GPUtest python=3.9
```

### Verified Configuration:

✅ Conda 25.1.1  
✅ Python 3.11.7 (also tested with 3.9)  
✅ CUDA 12.4.0  
✅ cuDNN 8.9.7.29  
✅ PyTorch 2.5.1 with GPU support (`pytorch-cuda=12.4`)  
✅ Visual Studio 2022 Community (no plugins)

Install PyTorch with GPU support:

```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

---

## Useful Resources

- [PyTorch Release Notes](https://github.com/pytorch/pytorch/blob/main/RELEASE.md)  
- [cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)  
- [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)  
- [Video Tutorial](https://www.youtube.com/watch?v=r7Am-ZGMef8)

---

## Additional Information (April 25th, 2025)

Tests performed on **RTX 3050TI with Ryzen 7 5800H**:

- **CUDA:** 12.6.3  
- **cuDNN:** 9.5.1.17  
- **Python:** 3.13  
- **PyTorch:** 2.7 (installed via pip)  
- **Conda:** 25.1.1  
- **C++:** C++17  

### `nvidia-smi` Output:

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