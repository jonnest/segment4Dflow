import torch
has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
# device = 'mps' if has_mps else 'cuda' if has_gpu else 'cpu'
device = 'cpu'
print("Running on ", device)
import sys
import torchvision
from torchvision import transforms as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import torchvision.utils as vutils

import numpy as np
import random
from matplotlib import pyplot as plt
import os,shutil, sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors

import scipy
import h5py
from matplotlib.colors import ListedColormap
import time
from skimage.measure import label   
import skimage as ski
import pyvista as pv
from scipy import ndimage
from skimage import io, img_as_float
from skimage.filters import frangi

# from medpy.filter import smoothing

if device=='mps':
    sys.path.append('/Users/jonathan/Documents/code/segment4DFlow/medformer3D')
else:
    sys.path.append('/home/jnestmann/code/segment4DFlow/medformer3D')
    
from unet import UNet
from attention_unet import AttentionUNet

from myFunctions.data_handling import *

# Reproducibility?
random.seed(10)
np.random.seed(10)
torch.manual_seed(10)

# data_dir = '/mnt/itetstor/jnestmann/mritrans_ibtscratch/nestmann/unet'
data_dir = '/mnt/itetstor/jnestmann/digitaltwin_ibtscratch/nestmann/unet'
w_dir = '/home/jnestmann/applications/unet'

test_folder     = 'test_c2_all' # 'images_volunteer' # 'test_c2_all'
model_dir       = 'model51'
model_type      = 'unet' # 'attention', 'unet'
ref_epoch       = 590
nr_base_filters = 32 # 32
res_folder      = f"results_{model_dir}_ep{ref_epoch}"
n_input         = 2
channels        = ['magnitude','cd']

test_path  = os.path.join(data_dir, test_folder)
model_path = os.path.join(w_dir, 'output')
res_path   = os.path.join(w_dir, 'results', res_folder)

model_name = 'SegmentBrainVessels'

# patchwise inference
# n_patch    = 64  # 64 
# n_overlap  = 8

# cropping in z direction
crop_z = 10
# use magnitude for thresholding
use_mag = False

# number of vessels/entities to extract
nr_vessels = 8
# number of dilations and erosions, subsequently applied
nr_dilate = 0
nr_erode = 0  

scalars = 'data'

# kernel size of 3d unet
kernel_size= [3,3,3,3,3]
scale = [2,2,2,2]

print("Model dir       ", model_dir)
print("Model type      ", model_type)
print("#base filters   ", nr_base_filters)

if not os.path.exists(res_path):
    os.makedirs(res_path)

if model_type=='unet':
    model = UNet(n_input,base_ch=nr_base_filters,scale=scale, kernel_size=kernel_size, num_classes=1, block='ConvNormAct', pool=True, norm='bn').to(torch.float).to(device)
elif model_type=='attention':
    model = AttentionUNet(n_input,base_ch=nr_base_filters,scale=scale, kernel_size=kernel_size, block='ConvNormAct', pool=True).to(torch.float).to(device)

checkpoint = torch.load(model_path + f'/{model_dir}/'+model_name+'_%05d.pt' %ref_epoch, map_location = device)
model.load_state_dict(checkpoint['model_state_dict'])
mySigmoid   = torch.nn.Sigmoid()

# walk through patients
patients = list_directories(test_path)

dice_scores = []

# Format : [patient,channel,slice,X,Y]
for patient_sel in patients:
    t0 = time.time()
    print("\nRead                   ", patient_sel)
    image1 = pv.read(os.path.join(test_path,patient_sel, f'{channels[0]}_'+patient_sel+'.vtk'))
    image2 = pv.read(os.path.join(test_path,patient_sel, f'{channels[1]}_'+patient_sel+'.vtk'))
    
    image1 = np.reshape(image1['data'], image1.dimensions, order='F')
    image2 = np.reshape(image2['data'], image2.dimensions, order='F')

    print("Image 1 shape          ", image1.shape)
    print("Image 2 shape          ", image2.shape)

    # for ii in range(image1.shape[-1]):
    #     image1[...,ii] = normalize_and_standardize(image1[...,ii])        
    #     image2[...,ii] = normalize_and_standardize(image2[...,ii])     
    image1 = normalize_and_standardize(image1)        
    image2 = normalize_and_standardize(image2)    

    # image1 = torch.tensor(image1, device=device).float()
    # image2 = torch.tensor(image2, device=device).float()
    
    test_image = np.stack((image1, image2), axis=0)
    print("Shape of test image    ", test_image.shape)
    
    with torch.no_grad():
        inf_input = np.expand_dims(test_image,0)
        # inf_input = inf_input[..., 30:250, 50:400, 30:170]
        # inf_input = inf_input[...,20:180]

        print("Shape of inf input     ", inf_input.shape)
        
        mask = create_mask1ch(mySigmoid(model(torch.from_numpy(inf_input).float().to(device)))).cpu().detach().numpy()[0,0,...]
        # mask = create_mask1ch(mySigmoid(model(torch.from_numpy(inf_input).float().to(device)))).cpu().detach().numpy()
    
    # mask = infer_3d_unet(model, volume=test_image, patch_size=(n_patch,n_patch,n_patch), overlap=(n_overlap,n_overlap,n_overlap), mySigmoid=mySigmoid).numpy()[0,...]
            
    # mask = mask.numpy()[0,...] 
    print("Shape of mask          ", mask.shape)

    # thresholding with magnitude
    if use_mag == True:
        mag = pv.read(os.path.join(test_path, "magnitude_"+patient_sel.split('_')[1]))
        mag = np.reshape(mag['data'], mag.dimensions, order='F')
        
        ### background noise through label
        labels = label(mag, background=0)
        print("Background size ", np.bincount(labels.flat)[0])
        mag_background_mask = labels == 0
        noise_level = np.mean(mag[mag_background_mask])
        
        mask = mask*mag
        mask[mask<=noise_level] = 0
        print("noise level ", noise_level)
        
    ### cropping
    # mask = mask[...,z_pixel:-(z_pixel-1)]
    # or 
    if crop_z != 0:
        mask[...,:crop_z] = 0
        mask[...,-(crop_z-1):] = 0
    
    ### filtering
    # mask = smoothing.anisotropic_diffusion(mask, niter=1, kappa=50, gamma=0.1, voxelspacing=None, option=2)
    # mask = smoothing.anisotropic_diffusion(mask, niter=1, kappa=20, gamma=0.1, voxelspacing=None, option=2)
    
    ### transform to binary mask
    mask[mask>0]=1
    mask[mask<=0]=0
    
    ### loop of dilation and erosions
    for i in range(nr_dilate):
        mask = ndimage.binary_dilation(mask).astype(int)
    for i in range(nr_erode):
        mask = ndimage.binary_erosion(mask).astype(int)

    # save full segmentation as vtk
    grid = pv.ImageData()
    grid.dimensions = [x_ for x_ in mask.shape]
    grid['data'] = np.reshape(mask, -1, order='F')
    grid.save(res_path+ '/' + f'segmentation_ep{ref_epoch}_{patient_sel}.vtk')

    contour = grid.contour(isosurfaces=[0.5], scalars='data')    
    connected_regions = contour.connectivity()
    
    print("number of region ids  ", connected_regions['RegionId'].max())
    connected_regions.save(res_path+ '/' + f'cc_ep{ref_epoch}_{patient_sel}.vtk')
    
    # for id in np.arange(nr_vessels+3):
    #     mask_i = connected_regions.threshold(value=(-0.5+id,0.5+id), scalars='RegionId')
    #     mask_i.save(res_path+ '/' + f'seg_id{id}_ep{ref_epoch}_{patient_sel}.vtk')
    
    ### labeled mask
    label_mask = getLargestSegmentations(mask, nr_vessels)
    grid = pv.ImageData()
    grid.dimensions = [x_ for x_ in label_mask.shape]
    grid['data'] = np.reshape(label_mask, -1, order='F')
    grid.save(res_path+ '/' + f'label_mask_ep{ref_epoch}_{patient_sel}.vtk')
    
    ### Inference end
    t1 = time.time()
    print("Inference time  ", round(t1-t0),3, " s")