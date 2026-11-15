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
from medpy.filter import smoothing

from skimage import io, img_as_float
from skimage.filters import frangi
from data_handling import *

sys.path.append('/Users/jonathan/Documents/code/segment4DFlow/medformer3D')
from unet import UNet as UNet_medformer
from attention_unet import AttentionUNet

# out_folder = "results_n1"
out_folder = "results3d_"

# model_dir  = 'Model_2D_n2'
model_dir  = 'Model_3D_n1'

# Models
# n1 : epoch 150 
# n2 : epoch 200
# n3 : epoch 110
ref_epoch   = 40

# cropping in z direction
crop_z = 0
# use magnitude for thresholding
use_mag = False
# number of input channels
input_channels = 2
# number of vessels/entities to extract
nr_vessels = 8
# number of dilations and erosions, subsequently applied
nr_dilate = 0
nr_erode = 0

test_path        = '/Users/jonathan/Documents/applications/unet/test_c1'
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
# [patient,channel,slice,X,Y]
mySigmoid   = torch.nn.Sigmoid()

patients = np.sort(next(os.walk(test_path))[2])
exclude_chars = ["magnitude"]
patients = [pat for pat in patients if not any(char in pat for char in exclude_chars)]

dice_scores = []
for patient_sel in patients:
    t0 = time.time()
    print("\nRead            ", patient_sel)
    
    image_sel = pv.read(os.path.join(test_path,patient_sel))
    image_sel = np.reshape(image_sel['data'], image_sel.dimensions, order='F')

    # normalize input image
    for ii in range(image_sel.shape[-1]): 
        image_sel[...,ii] = normalize_and_standardize(image_sel[...,ii])
        
    nn_masks = np.zeros_like(image_sel)
    print("Shape of mask   ", nn_masks.shape)
    
    with torch.no_grad():
        # loop over z-dir, prediction of circles most feasible
        for i in range(image_sel.shape[-1]):
            # print("shape inference input  ", np.expand_dims(np.expand_dims(image_sel[...,i],0),0).shape)
            mask = create_mask1ch(mySigmoid(model(torch.from_numpy(np.expand_dims(np.expand_dims(image_sel[...,i],0),0)).float().to(device)))).cpu().detach().numpy()[:,0,...]
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
    connected_regions.save(out_path+ '/' + f'cc_ep{ref_epoch}_{patient_sel}')
    
    # for id in np.arange(nr_vessels+3):
    #     mask_i = connected_regions.threshold(value=(-0.5+id,0.5+id), scalars='RegionId')
    #     mask_i.save(out_path+ '/' + f'seg_id{id}_ep{ref_epoch}_{patient_sel}')
    
    ### labeled mask
    label_mask = getLargestSegmentations(mask, nr_vessels)
    grid = pv.ImageData()
    grid.dimensions = [x_ for x_ in label_mask.shape]
    grid['data'] = np.reshape(label_mask, -1, order='F')
    grid.save(out_path+ '/' + f'label_mask_ep{ref_epoch}_{patient_sel}')
    
    ### Inference end
    t1 = time.time()
    print("Inference time  ", round(t1-t0),3, " s")
    
    
    # ------------------------------------------------------------------
    ### Calculate sizes of each labeled region
    labels = label(mask, connectivity=1)
    # Assume at least 1 connected component
    assert(labels.max() != 0) 
    # Count the size of each component, excluding the background (label 0)
    component_sizes = np.bincount(labels.flat)[1:] #
    print("component sizes ", component_sizes)
    # [::-1] sort by ascending order
    largest_labels = np.argsort(component_sizes)[-nr_vessels:][::-1] + 1  # +1 to adjust for background exclusion
    print("largest_labels ", largest_labels)

    for i in range(nr_vessels):
        print("Loop ", i, "    - label  ", largest_labels[i])
        ### Apply the mask to the original segmentation
        # segmentation_mask = np.isin(labels, [largest_labels[id]])
        segmentation_mask = labels == largest_labels[i]
        # import pdb; pdb.set_trace()
        
        grid = pv.ImageData()
        grid.dimensions = [x_ for x_ in segmentation_mask.shape]
        grid['data'] = np.reshape(segmentation_mask,-1,order='F')
        grid.save(out_path+ '/' + f'seg_ep{ref_epoch}_id{i}_{patient_sel}')
        
        
    # ------------------------------------------------------------------
    '''
    # ### frangi filter
    # ### Apply the Frangi filter
    # frangi_mask = frangi(image_sel, scale_range=None, scale_step=2, alpha=0.5, beta=0.5, black_ridges=True)

    # ### save segmentation as vtk
    # grid = pv.ImageData()
    # grid.dimensions = [x_ for x_ in frangi_mask.shape]
    # grid['data'] = np.reshape(frangi_mask,-1,order='F')
    # # grid.save(out_path+ '/' + f'frangi_seg_ep{ref_epoch}_{patient_sel}')
    # grid.save(out_path+ '/' + f'frangi_{patient_sel}')
    # ### Frangi end
    # t2 = time.time()
    # print("Frangi filter time  ", round(t2-t1),3, " s")
    '''
    
    # ------------------------------------------------------------------
    '''
    # ---------------- PyVista --------------
    ### save every entity 
    grid = pv.ImageData()
    grid.dimensions = [x_ for x_ in mask.shape]
    grid['data'] = np.reshape(np.round(mask).astype('double'), -1, order='F')
    # Convert ImageData to UnstructuredGrid
    ugrid = grid.cast_to_unstructured_grid()
    # Apply connectivity filter to get connected components (cc)
    cc = ugrid.connectivity()

    # grid = grid.connectivity()
    cc.save(out_path + '/' + f'seg_cc_ep{ref_epoch}_{patient_sel}')
    
    import pdb; pdb.set_trace()
        
    
    # for id in range(nr_vessels):
    #     print("Loop id ", id)
    #     # get largest connected component
    #     # largest_cc = grid.connectivity()
        
    for id in np.arange(grid['RegionId'].max()+1):
        # threshold of scalar value for natural number 0,1,2,3,..
        temp = grid.threshold(value=(-0.5+id,0.5+id), scalars='RegionId')
        
        grid.save(out_path + '/' + f'seg_ep{ref_epoch}_id{id}_{patient_sel}')

        
        # mask_grid = np.reshape(grid['data'], grid.dimensions, order='F')
        # mask_largest = np.reshape(largest_cc.point_data['RegionId'], grid.dimensions, order='F')
        
        pv_rest = pv.ImageData()
        pv_rest.dimensions = [x_ for x_ in mask.shape]
        pv_rest['data'] = np.reshape(mask_grid - mask_largest, -1, order='F')
        pv_rest.save(out_path + '/' + f'rest_mask_ep{ref_epoch}_id{id}_{patient_sel}')
        
        # rest_mask = grid.boolean_difference(largest_cc)
        # rest_mask.save(out_path + '/' + f'rest_mask_ep{ref_epoch}_id{id}_{patient_sel}')
        
        import pdb; pdb.set_trace()
        
        break
        # # find the largest connected component
        # labels = grid.connectivity()['RegionId'] #.astype(int)
        # print("labels ", labels)
        # unique_labels, counts = np.unique(labels, return_counts=True)
        # print("unique_labels ", unique_labels)
        
        # largest_label = unique_labels[np.argmax(counts)]
        # print("Largest label ", largest_label)
        
        # save largest component
        # pv_mask.save(out_path + '/' + f'seg_ep{ref_epoch}_id{id}_{patient_sel}')
        # largest_component = grid.connectivity().threshold(value=largest_label, scalars="RegionId")
        # largest_component.save(out_path + '/' + f'seg_ep{ref_epoch}_id{id}_{patient_sel}')
        
        # exclude largest label from grid
        # grid = grid.connectivity().threshold(value=largest_label, invert=True, scalars="RegionId")
        # grid.save(out_path + '/' + f'grid_ep{ref_epoch}_id{id}_{patient_sel}')
    # --------------------------------------
    '''
    