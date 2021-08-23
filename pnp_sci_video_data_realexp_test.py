# %% [markdown]
# ## PnP-TV-FastDVDNet for real data experiment
# ### Code credit
# [Xin Yuan](https://www.bell-labs.com/usr/x.yuan "Dr. Xin Yuan, Bell Labs"), [Bell Labs](https://www.bell-labs.com/), xyuan@bell-labs.com, created Aug 7, 2018.  
# [Yang Liu](https://liuyang12.github.io "Yang Liu, MIT"), [MIT](https://www.mit.edu/), yliu12@mit.edu, updated Jan 20, 2019.
# [Zhihong Zhang](https://zhihongz.github.io/ "Zhihong Zhang, Tsinghua University"), [Tsinghua University](http://www.tsinghua.edu.cn/publish/thu2018en/index.html), zhangzh19@mails.tsinghua.edu.cn, updated Mar 19, 2021.

# %%
import os
from os.path import join as opj
import time
import math
import h5py
import numpy as np
import scipy.io as sio
from scipy.io.matlab.mio import _open_file
from scipy.io.matlab.miobase import get_matfile_version
import matplotlib.pyplot as plt
from statistics import mean
from datetime import datetime

from dvp_linear_inv import admmdenoise_cacti
from joint_dvp_linear_inv import joint_admmdenoise_cacti

from utils import (A_, At_, show_n_save_res)
import torch
from packages.ffdnet.models import FFDNet
from packages.fastdvdnet.models import FastDVDnet

# %%
# [0] environment configuration

## flags and params
save_res_flag = 1           # save results
show_res_flag = 1           # show results
save_param_flag = 1        # save params
# test_algo_flag = ['all']		# choose algorithms: 'all', 'gaptv', 'gapffdnet','gaptv+fastdvdnet'
test_algo_flag = ['gaptv+fastdvdnet']

datasetdir = './dataset/real_data' # data dir
resultsdir = './results' # results dir  # results
datname = 'data_E_478_20210315_roi400-1440_sz320_Cr6'


matfile = datasetdir + '/' + datname + '.mat'  # path of the .mat data file


# %%
# [1] load data
# MATLAB .mat v7.2 or lower versions
if get_matfile_version(_open_file(matfile, appendmat=True)[0])[0] < 2:
    # for '-v7.2' and lower version of .mat file (MATLAB)
    file = sio.loadmat(matfile)
    order = 'K'  # [order] keep as the default order in Python/numpy
    meas = np.float32(file['meas'], order=order)
    mask = np.float32(file['mask'], order=order)
    # orig = np.float32(file['orig'], order=order)
else:  # MATLAB .mat v7.3
    file = h5py.File(matfile, 'r')  # for '-v7.3' .mat file (MATLAB)
    order = 'F'  # [order] switch to MATLAB array order
    meas = np.float32(file['meas'], order=order).transpose()
    mask = np.float32(file['mask'], order=order).transpose()
    # orig = np.float32(file['orig'], order=order).transpose()
orig = None  # no orig - ground truthcrop_
# print(meas.shape, mask.shape, orig.shape)


mask_sum = np.sum(mask, axis=2)
mask_sum[mask_sum == 0] = 1

# zzh: expand dim for a single 'meas'
if meas.ndim < 3:
    meas = np.expand_dims(meas, 2)
    # print(meas.shape)
# print('meas, mask, orig:', meas.shape, mask.shape, orig.shape)

# normalize data
mask_max = np.max(mask)
mask = mask/mask_max
meas = meas/mask_max

# --------- param ------------
iframe = 0
nframe = 1
nmask = mask.shape[2]

# MAXB = 255. # for 8bit
# MAXB = 65535. # for 16bit
MAXB = 65535/nmask  # real measurement's data range is Cr times less then simulated measment
# print(MAXB)
# --------- param ------------

# nframe = meas.shape[2]


# common parameters and pre-calculation for PnP
# define forward model and its transpose
def A(x): return A_(x, mask)  # forward model function handle
def At(y): return At_(y, mask)  # transpose of forward model


# %%
# [2.1]  GAP-TV
if ('all' in test_algo_flag) or ('gaptv' in test_algo_flag):
    projmeth = 'gap'  # projection method
    _lambda = 1  # regularization factor, [original set]
    accelerate = True  # enable accelerated version of GAP
    denoiser = 'tv'  # total variation (TV)
    iter_max = 20  # maximum number of iterations
    # tv_weight = 0.25 # TV denoising weight (larger for smoother but slower) [kobe:0.25; ]
    tv_weight = 1  # TV denoising weight
    tv_iter_max = 5  # TV denoising maximum number of iterations each
    vgaptv, tgaptv, psnr_gaptv, ssim_gaptv, psnrall_gaptv = admmdenoise_cacti(meas, mask, A, At,
                                                                              projmeth=projmeth, v0=None, orig=orig,
                                                                              iframe=iframe, nframe=nframe,
                                                                              MAXB=MAXB, maskdirection='plain',
                                                                              _lambda=_lambda, accelerate=accelerate,
                                                                              denoiser=denoiser, iter_max=iter_max,
                                                                              tv_weight=tv_weight,
                                                                              tv_iter_max=tv_iter_max)

    print('-'*20+'\n{}-{} running time {:.1f} seconds.\n'.format(
        projmeth.upper(), denoiser.upper(),  tgaptv)+'-'*20)
    show_n_save_res(vgaptv, tgaptv, psnr_gaptv, ssim_gaptv, psnrall_gaptv, orig, nmask, resultsdir,
                    projmeth+denoiser+'_'+datname+datetime.now().strftime('@T%Y%m%d-%H-%M'), iframe=iframe, nframe=nframe, MAXB=MAXB,
                    show_res_flag=show_res_flag, save_res_flag=save_res_flag,
                    tv_weight=tv_weight, iter_max=iter_max)

# %%
# [2.2] GAP-FFDNet (FFDNet-based frame-wise video denoising)
if ('all' in test_algo_flag) or ('gapffdnet' in test_algo_flag):
    projmeth = 'gap'  # projection method
    _lambda = 1  # regularization factor, [original set]
    # _lambda = 1.5
    accelerate = True  # enable accelerated version of GAP
    denoiser = 'ffdnet'  # video non-local network
    noise_estimate = False  # disable noise estimation for GAP
    sigma = [50/255, 25/255, 12/255, 6/255]  # pre-set noise standard deviation
    iter_max = [10, 10, 10, 10]  # maximum number of iterations
    # sigma    = [12/255, 6/255] # pre-set noise standard deviation
    # iter_max = [10,10] # maximum number of iterations
    useGPU = True  # use GPU

    # pre-load the model for FFDNet image denoising
    in_ch = 1
    model_fn = 'packages/ffdnet/models/net_gray.pth'
    # Absolute path to model file
    # model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), model_fn)

    # Create model
    net = FFDNet(num_input_channels=in_ch)
    # Load saved weights
    if useGPU:
        state_dict = torch.load(model_fn)
        device_ids = [0]
        model = torch.nn.DataParallel(net, device_ids=device_ids).cuda()
    else:
        state_dict = torch.load(model_fn, map_location='cpu')
        # CPU mode: remove the DataParallel wrapper
        state_dict = remove_dataparallel_wrapper(state_dict)
        model = net
    model.load_state_dict(state_dict)
    model.eval()  # evaluation mode

    vgapffdnet, tgapffdnet, psnr_gapffdnet, ssim_gapffdnet, psnrall_gapffdnet = admmdenoise_cacti(meas, mask, A, At,
                                                                                                  projmeth=projmeth, v0=None, orig=orig,
                                                                                                  iframe=iframe, nframe=nframe,
                                                                                                  MAXB=MAXB, maskdirection='plain',
                                                                                                  _lambda=_lambda, accelerate=accelerate,
                                                                                                  denoiser=denoiser, model=model,
                                                                                                  iter_max=iter_max, sigma=sigma)

    print('-'*20+'\n{}-{} running time {:.1f} seconds.\n'.format(
        projmeth.upper(), denoiser.upper(), tgapffdnet)+'-'*20)
    show_n_save_res(vgapffdnet, tgapffdnet, psnr_gapffdnet, ssim_gapffdnet, psnrall_gapffdnet, orig, nmask, resultsdir,
                    projmeth+denoiser+'_'+datname+datetime.now().strftime('@T%Y%m%d-%H-%M'), iframe=iframe, nframe=nframe, MAXB=MAXB,
                    show_res_flag=show_res_flag, save_res_flag=save_res_flag,
                    iter_max=iter_max, sigma=sigma)

# %%
# [2.3] GAP-TV+FASTDVDNET
if ('all' in test_algo_flag) or ('gaptv+fastdvdnet' in test_algo_flag):
    projmeth = 'gap'  # projection method
    _lambda = 1  # regularization factor, [original set]
    accelerate = True  # enable accelerated version of GAP
    denoiser = 'tv+fastdvdnet'  # video non-local network
    noise_estimate = False  # disable noise estimation for GAP
    sigma1 = [0]  # pre-set noise standard deviation for 1st period denoise
    iter_max1 = 10  # maximum number of iterations for 1st period denoise
    # pre-set noise standard deviation for 2nd period denoise
    sigma2 = [150/MAXB, 80/MAXB, 50/MAXB, 30/MAXB]
    # maximum number of iterations for 2nd period denoise
    iter_max2 = [60, 60, 60, 60]
    tv_iter_max = 5  # TV denoising maximum number of iterations each
    # TV denoising weight (larger for smoother but slower) [kobe:0.25]
    tv_weight = 12
    # tv_weight = 0.5 # TV denoising weight (larger for smoother but slower) [kobe:0.25]
    tvm = 'tv_chambolle'
    # sigma    = [12/255] # pre-set noise standard deviation
    # iter_max = [20] # maximum number of iterations
    useGPU = True  # use GPU

    # pre-load the model for fastdvdnet image denoising
    NUM_IN_FR_EXT = 5  # temporal size of patch
    model = FastDVDnet(num_input_frames=NUM_IN_FR_EXT, num_color_channels=1)

    # Load saved weights
    state_temp_dict = torch.load('./packages/fastdvdnet/model_gray.pth')
    if useGPU:
        device_ids = [0]
        # model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
        model = model.cuda()
    # else:
        # # CPU mode: remove the DataParallel wrapper
        # state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)

    model.load_state_dict(state_temp_dict)

    # Sets the model in evaluation mode (e.g. it removes BN)
    model.eval()

    vgaptvfastdvdnet, tgaptvfastdvdnet, psnr_gaptvfastdvdnet, ssim_gaptvfastdvdnet, psnrall_gaptvfastdvdnet = joint_admmdenoise_cacti(meas, mask, A, At,
                                                                                                                                      projmeth=projmeth, v0=None, orig=orig,
                                                                                                                                      iframe=iframe, nframe=nframe,
                                                                                                                                      MAXB=MAXB, maskdirection='plain',
                                                                                                                                      _lambda=_lambda, accelerate=accelerate,
                                                                                                                                      denoiser=denoiser, iter_max1=iter_max1, iter_max2=iter_max2,
                                                                                                                                      tv_weight=tv_weight, tv_iter_max=tv_iter_max,
                                                                                                                                      model=model, sigma1=sigma1, sigma2=sigma2, tvm=tvm)

    print('-'*20+'\n{}-{} running time {:.1f} seconds.\n'.format(
        projmeth.upper(), denoiser.upper(), tgaptvfastdvdnet)+'-'*20)
    show_n_save_res(vgaptvfastdvdnet, tgaptvfastdvdnet, psnr_gaptvfastdvdnet, ssim_gaptvfastdvdnet, psnrall_gaptvfastdvdnet, orig, nmask, resultsdir,
                    projmeth+denoiser+'_'+datname+datetime.now().strftime('@T%Y%m%d-%H-%M'), iframe=iframe, nframe=nframe, MAXB=MAXB,
                    show_res_flag=show_res_flag, save_res_flag=save_res_flag, tv_iter_max=tv_iter_max,
                    tv_weight=tv_weight, iter_max1=iter_max1, iter_max2=iter_max2, sigma1=sigma1, sigma2=sigma2)
# %%
# [4] show res
if show_res_flag:
    plt.show()

# %%
# [5] save params
if save_param_flag:
    # params path
    param_path = resultsdir+'/savedfig/param_finetune.txt'
    if os.path.exists(param_path):
        writemode = 'a+'
    else:
        writemode = 'w'
    with open(param_path, writemode) as f:
        # customized contents
        f.write(projmeth+denoiser+'_'+datname +
                datetime.now().strftime('@T%Y%m%d-%H-%M')+':\n')
        f.write('\titer_max1 = ' + str(iter_max1) + '\n')
        f.write('\tsigma2 = ' + str([255*x for x in sigma2]) + '/255\n')
        f.write('\titer_max2 = ' + str(iter_max2) + '\n')
        f.write('\ttv_iter_max = ' + str(tv_iter_max) + '\n')
        f.write('\ttv_weight = ' + str(tv_weight) + '\n')
