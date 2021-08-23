# 10 Mega Pixel Snapshot Compressive Imaging with A Hybrid Coded Aperture (HCA-SCI) 
[Zhihong Zhang](https://zhihongz.github.io/), Tsinghua University, z_zhi_hong@163.com, updated Mar 19, 2021 

This repository contains the Python (PyTorch) code for the paper **10 Mega Pixel Snapshot Compressive Imaging with A Hybrid Coded Aperture**

The initial Python code for [PnP-SCI](https://github.com/liuyang12/PnP-SCI) was from [Yang Liu](https://liuyang12.github.io "Yang Liu, MIT") and [Dr. Xin Yuan](https://sites.google.com/site/eiexyuan/ "Dr. Xin Yuan, Bell Labs").


## How to run this code
This code is tested on Windows 10 CUDA 10.0.130, CuDNN 7.6.0, and PyTorch 1.2.0. It is supposed to work on other platforms (Linux or Windows) with CUDA-enabled GPU(s). 

1. Put the dataset into `./dataset`.
2. Create the virtual environment with required Python packages via  
`conda env create -f environment.yml`
2. Run  `pnp_sci_video_data_simuexp_test.py` to test the simulated data.
3. Run  `pnp_sci_video_data_realexp_test.py` to test the real data.


## More information
- Please refer to [PnP-SCI](https://github.com/liuyang12/PnP-SCI) for detailed information about PnP-SCI algorithm.
- The code for BIRNAT can be find in https://github.com/BoChenGroup/BIRNAT

## Citation
```

```