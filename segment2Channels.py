import torch
has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = 'mps' if has_mps else 'gpu' if has_gpu else 'cpu'
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

from data_handling import create_mask1ch, dice_score, getPatients, normalize_and_standardize

import matplotlib.pyplot as plt
from matplotlib import colors

from unet import UNet as UNet_medformer
from attention_unet import AttentionUNet
import scipy
import h5py
from matplotlib.colors import ListedColormap
import time
from skimage.measure import label   
import skimage as ski
import pyvista as pv
from scipy import ndimage
from medpy.filter import smoothing

from skimage import io, img_as_float
from skimage.filters import frangi

from data_handling import *

out_folder = "results_n3"
model_dir  = 'Model_2D_n3'

# Models
# n1 : epoch 150 
# n2 : epoch 200 
ref_epoch   = 110 # 110, 150, 200

# cropping in z direction
crop_z = 5
# use magnitude for thresholding
use_mag = False
# number of input channels
input_channels = 2
channel1 = 'magnitude'
channel2 = 'velocity'
# number of vessels/entities to extract
nr_vessels = 8
# number of dilations and erosions, subsequently applied
nr_dilate = 0
nr_erode = 0  

test_path        = '/Users/jonathan/Documents/applications/unet/test_c2'
model_path       = '/Users/jonathan/Documents/applications/unet/output'
out_path         = f'/Users/jonathan/Documents/applications/unet/{out_folder}'

model_name = 'SegmentBrainVessels'

scalars = 'data'
# Reproducibility?
random.seed(10)
np.random.seed(10)
torch.manual_seed(10)




if not os.path.exists(out_path):
    os.makedirs(out_path)
        
model = AttentionUNet(input_channels,1,64,block='SingleConv',pool=True).to(torch.float).to(device)
checkpoint = torch.load(model_path + f'/{model_dir}/'+model_name+'_%05d.pt' %ref_epoch, map_location = device)
model.load_state_dict(checkpoint['model_state_dict'])
mySigmoid   = torch.nn.Sigmoid()

# walk through patients
patients = list_directories(test_path)

dice_scores = []

# Format : [patient,channel,slice,X,Y]
for patient_sel in patients:
    t0 = time.time()
    print("\nRead            ", patient_sel)
    
    image1 = pv.read(os.path.join(test_path,patient_sel, f'{channel1}_'+patient_sel+'.vtk'))
    image2 = pv.read(os.path.join(test_path,patient_sel, f'{channel2}_'+patient_sel+'.vtk'))
    
    image1 = np.reshape(image1['data'], image1.dimensions, order='F')
    image2 = np.reshape(image2['data'], image2.dimensions, order='F')

    for ii in range(image1.shape[-1]):
        image1[...,ii] = normalize_and_standardize(image1[...,ii])        

    for ii in range(image2.shape[-1]):
        image2[...,ii] = normalize_and_standardize(image2[...,ii])     
        
    test_image = np.stack((image1, image2), axis=0)
    print("\nShape of test image    ", test_image.shape)
    
    # output masks same as one input image
    nn_masks = np.zeros_like(image1)
    print("\nShape of mask          ", nn_masks.shape)
    
    # inference input : (1, #input_channels, #X, #Y)
    print("\nShape inference input  ", np.expand_dims(test_image[...,0],0).shape)
    with torch.no_grad():
        # loop over z-dir, prediction of circles most feasible
        for i in range(test_image.shape[-1]):
            inf_input = np.expand_dims(test_image[...,i],0)
            mask = create_mask1ch(mySigmoid(model(torch.from_numpy(inf_input).float().to(device)))).cpu().detach().numpy()[:,0,...]
            nn_masks[...,i] = mask[0,...]
    
    mask = nn_masks.copy()
    
    ### thresholding with magnitude
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
    grid.save(out_path+ '/' + f'segmentation_ep{ref_epoch}_{patient_sel}.vtk')

    contour = grid.contour(isosurfaces=[0.5], scalars='data')    
    connected_regions = contour.connectivity()
    
    print("number of region ids  ", connected_regions['RegionId'].max())
    connected_regions.save(out_path+ '/' + f'cc_ep{ref_epoch}_{patient_sel}.vtk')
    
    # for id in np.arange(nr_vessels+3):
    #     mask_i = connected_regions.threshold(value=(-0.5+id,0.5+id), scalars='RegionId')
    #     mask_i.save(out_path+ '/' + f'seg_id{id}_ep{ref_epoch}_{patient_sel}.vtk')
    
    ### labeled mask
    label_mask = getLargestSegmentations(mask, nr_vessels)
    grid = pv.ImageData()
    grid.dimensions = [x_ for x_ in label_mask.shape]
    grid['data'] = np.reshape(label_mask, -1, order='F')
    grid.save(out_path+ '/' + f'label_mask_ep{ref_epoch}_{patient_sel}.vtk')
    
    ### Inference end
    t1 = time.time()
    print("Inference time  ", round(t1-t0),3, " s")