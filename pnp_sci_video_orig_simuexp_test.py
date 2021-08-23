# %% [markdown]
# ## PnP-TV-FastDVDNet for simulated data experiment
# ### Code credit
# [Xin Yuan](https://www.bell-labs.com/usr/x.yuan "Dr. Xin Yuan, Bell Labs"), [Bell Labs](https://www.bell-labs.com/), xyuan@bell-labs.com, created Aug 7, 2018.  
# [Yang Liu](https://liuyang12.github.io "Yang Liu, MIT"), [MIT](https://www.mit.edu/), yliu12@mit.edu, updated Jan 20, 2019.
# [Zhihong Zhang](https://zhihongz.github.io/ "Zhihong Zhang, Tsinghua University"), [Tsinghua University](http://www.tsinghua.edu.cn/publish/thu2018en/index.html), zhangzh19@mails.tsinghua.edu.cn, updated Mar 19, 2021.

# %%
import os
import time
import math
import h5py
import numpy as np
import scipy.io as sio
from statistics import mean
from dvp_linear_inv import admmdenoise_cacti
from joint_dvp_linear_inv import joint_admmdenoise_cacti
from utils import (A_, At_, show_n_save_res)
import matplotlib.pyplot as plt
from scipy.io.matlab.mio import _open_file
from scipy.io.matlab.miobase import get_matfile_version
# import deep models
import torch
from packages.ffdnet.models import FFDNet
from packages.fastdvdnet.models import FastDVDnet



# %%
# [0] environment configuration
## [0.1] path and data name
orig_dir = './dataset/simu_data/orig'
mask_dir = './dataset/simu_data/mask' # mask dataset
resultsdir = './results' # results dir

orig_name = 'football_256'    

mask_name = 'shift_binary_mask_256_10f'    # name of 'mask'


origpath = orig_dir + '/' + orig_name + '.mat' # path of the .mat orig file
maskpath = mask_dir + '/' + mask_name + '.mat' # path of the .mat mask file


## [0.2] flags and params
show_res_flag = 1           # show results
save_res_flag = 1          # save results
# choose algorithms: 
# 'all', 'gaptv', 'gapffdnet',  'gaptv+fastdvdnet'
# test_algo_flag = ['all']	
test_algo_flag = ['gaptv+fastdvdnet']	

# noise
gaussian_noise_level = 5
poisson_noise = False


# %%
# [1] load data
if get_matfile_version(_open_file(maskpath, appendmat=True)[0])[0] < 2: # MATLAB .mat v7.2 or lower versions
    origfile = sio.loadmat(origpath) # for '-v7.2' and below .mat file (MATLAB)
    maskfile = sio.loadmat(maskpath)
    orig = np.array(origfile['orig'])
    mask = np.array(maskfile['mask'])
    mask = np.float32(mask)
    orig = np.float32(orig)
else: # MATLAB .mat v7.3
    with h5py.File(origpath, 'r') as origfile: # for '-v7.3' .mat file (MATLAB)
        orig = np.array(origfile['orig'])
        orig = np.float32(orig).transpose((2,1,0))
        
    with h5py.File(maskpath, 'r') as maskfile: # for '-v7.3' .mat file (MATLAB)
        # print(list(file.keys()))
        mask = np.array(maskfile['mask'])
        mask = np.float32(mask).transpose((2,1,0))

#  calc meas
nmask = mask.shape[2]
norig = orig.shape[2]
meas = np.zeros([orig.shape[0], orig.shape[1], norig//nmask])
for i in range(norig//nmask):
    tmp_orig = orig[:,:,i*nmask:(i+1)*nmask]
    meas[:,:,i] = np.sum(tmp_orig*mask, 2)

# zzh: expand dim for a single 'meas'
if meas.ndim<3:
    meas = np.expand_dims(meas,2)
    # print(meas.shape)
# print('meas, mask, orig:', meas.shape, mask.shape, orig.shape)

# add nosie
# print('* before add noise: orig {}  mask {} meas {}'.format(np.mean(orig), np.mean(mask), np.mean(meas)))
gaussian_noise = np.random.randn(meas.shape[0], meas.shape[1], meas.shape[2])*gaussian_noise_level
meas = meas + gaussian_noise
if poisson_noise:
    meas = np.random.poisson(meas)


# normalize data
mask_max = np.max(mask) 
mask = mask/mask_max
meas = meas/mask_max


  
iframe = 0
nframe = 1
MAXB = 255.

# common parameters and pre-calculation for PnP
# define forward model and its transpose
A  = lambda x :  A_(x, mask) # forward model function handle
At = lambda y : At_(y, mask) # transpose of forward model


# %%
## [2.1]  GAP-TV
if ('all' in test_algo_flag) or ('gaptv' in test_algo_flag):
    projmeth = 'gap' # projection method
    _lambda = 1 # regularization factor, [original set]
    accelerate = True # enable accelerated version of GAP
    denoiser = 'tv' # total variation (TV)
    iter_max = 100 # maximum number of iterations
    tv_weight = 0.25 # TV denoising weight (larger for smoother but slower) [kobe:0.25; ]
    tv_iter_max = 5 # TV denoising maximum number of iterations each

    vgaptv,tgaptv,psnr_gaptv,ssim_gaptv,psnrall_gaptv = admmdenoise_cacti(meas, mask, A, At,
                                            projmeth=projmeth, v0=None, orig=orig,
                                            iframe=iframe, nframe=nframe,
                                            MAXB=MAXB, maskdirection='plain',
                                            _lambda=_lambda, accelerate=accelerate,
                                            denoiser=denoiser, iter_max=iter_max, 
                                            tv_weight=tv_weight, 
                                            tv_iter_max=tv_iter_max)

    print('-'*20+'\n{}-{} PSNR {:2.3f} dB, SSIM {:.4f}, running time {:.1f} seconds.\n'.format(
        projmeth.upper(), denoiser.upper(), mean(psnr_gaptv), mean(ssim_gaptv), tgaptv)+'-'*20)
    show_n_save_res(vgaptv,tgaptv,psnr_gaptv,ssim_gaptv,psnrall_gaptv, orig, nmask, resultsdir, 
                        projmeth+denoiser+'_'+orig_name, iframe=iframe,nframe=nframe, MAXB=MAXB, 
                        show_res_flag=show_res_flag, save_res_flag=save_res_flag,
                        tv_weight=tv_weight, iter_max = iter_max)

# %%
## [2.2] GAP-FFDNet (FFDNet-based frame-wise video denoising)
if ('all' in test_algo_flag) or ('gapffdnet' in test_algo_flag):
    projmeth = 'gap' # projection method
    _lambda = 1 # regularization factor, [original set]
    # _lambda = 1.5
    accelerate = True # enable accelerated version of GAP
    denoiser = 'ffdnet' # video non-local network 
    noise_estimate = False # disable noise estimation for GAP
    sigma    = [50/255, 25/255, 12/255, 6/255] # pre-set noise standard deviation
    iter_max = [10, 10, 10, 10] # maximum number of iterations
    # sigma    = [12/255, 6/255] # pre-set noise standard deviation
    # iter_max = [10,10] # maximum number of iterations
    useGPU = True # use GPU

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
    model.eval() # evaluation mode

    vgapffdnet,tgapffdnet,psnr_gapffdnet,ssim_gapffdnet,psnrall_gapffdnet = admmdenoise_cacti(meas, mask, A, At,
                                            projmeth=projmeth, v0=None, orig=orig,
                                            iframe=iframe, nframe=nframe,
                                            MAXB=MAXB, maskdirection='plain',
                                            _lambda=_lambda, accelerate=accelerate,
                                            denoiser=denoiser, model=model, 
                                            iter_max=iter_max, sigma=sigma)

    print('-'*20+'\n{}-{} PSNR {:2.3f} dB, SSIM {:.4f}, running time {:.1f} seconds.\n'.format(
        projmeth.upper(), denoiser.upper(), mean(psnr_gapffdnet), mean(ssim_gapffdnet), tgapffdnet)+'-'*20)
    show_n_save_res(vgapffdnet,tgapffdnet,psnr_gapffdnet,ssim_gapffdnet,psnrall_gapffdnet, orig, nmask, resultsdir, 
                        projmeth+denoiser+'_'+orig_name, iframe=iframe,nframe=nframe, MAXB=MAXB, 
                        show_res_flag=show_res_flag, save_res_flag=save_res_flag,
                        iter_max = iter_max, sigma=sigma)



# %%
## [2.3] GAP-TV+FASTDVDNET
import torch
from packages.fastdvdnet.models import FastDVDnet

if ('all' in test_algo_flag) or ('gaptv+fastdvdnet' in test_algo_flag):
    projmeth = 'gap' # projection method
    _lambda = 1 # regularization factor, [original set]
    accelerate = True # enable accelerated version of GAP
    denoiser = 'tv+fastdvdnet' # video non-local network 
    noise_estimate = False # disable noise estimation for GAP
    sigma1    = [0] # pre-set noise standard deviation for 1st period denoise 
    iter_max1 = 10 # maximum number of iterations for 1st period denoise   
    sigma2    = [100/255, 50/255, 25/255] # pre-set noise standard deviation for 2nd period denoise 
    iter_max2 = [60, 100, 150] # maximum number of iterations for 2nd period denoise    
    # sigma2    = [50/255, 25/255] # pre-set noise standard deviation for 2nd period denoise 
    # iter_max2 = [20, 20] # maximum number of iterations for 2nd period denoise   
    tv_iter_max = 5 # TV denoising maximum number of iterations each
    tv_weight = 0.5 # TV denoising weight (larger for smoother but slower) [kobe:0.25]
    tvm = 'tv_chambolle'
    # sigma    = [12/255] # pre-set noise standard deviation
    # iter_max = [20] # maximum number of iterations
    useGPU = True # use GPU

    # pre-load the model for fastdvdnet image denoising
    NUM_IN_FR_EXT = 5 # temporal size of patch
    model = FastDVDnet(num_input_frames=NUM_IN_FR_EXT,num_color_channels=1)

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

    vgaptvfastdvdnet,tgaptvfastdvdnet,psnr_gaptvfastdvdnet,ssim_gaptvfastdvdnet,psnrall_gaptvfastdvdnet = joint_admmdenoise_cacti(meas, mask, A, At,
                                            projmeth=projmeth, v0=None, orig=orig,
                                            iframe=iframe, nframe=nframe,
                                            MAXB=MAXB, maskdirection='plain',
                                            _lambda=_lambda, accelerate=accelerate,
                                            denoiser=denoiser, iter_max1=iter_max1, iter_max2=iter_max2,
                                            tv_weight=tv_weight, tv_iter_max=tv_iter_max, 
                                            model=model, sigma1=sigma1, sigma2=sigma2, tvm=tvm)

    print('-'*20+'\n{}-{} PSNR {:2.3f} dB, SSIM {:.4f}, running time {:.1f} seconds.\n'.format(
        projmeth.upper(), denoiser.upper(), mean(psnr_gaptvfastdvdnet), mean(ssim_gaptvfastdvdnet), tgaptvfastdvdnet)+'-'*20)
    show_n_save_res(vgaptvfastdvdnet,tgaptvfastdvdnet,psnr_gaptvfastdvdnet,ssim_gaptvfastdvdnet,psnrall_gaptvfastdvdnet, orig, nmask, resultsdir, 
                        projmeth+denoiser+'_'+orig_name, iframe=iframe,nframe=nframe, MAXB=MAXB, 
                        show_res_flag=show_res_flag, save_res_flag=save_res_flag,
                        tv_weight=tv_weight, iter_max1=iter_max1, iter_max2=iter_max2, sigma1=sigma1, sigma2=sigma2)
          
# %%
# [4] show res
# if show_res_flag:
#     plt.show()
