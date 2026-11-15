import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
    print('I found a GPU')

import sys
sys.path.append('./medformer2D/')
sys.path.append('./myFunctions/')

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

from skimage.measure import label   

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

# Reproducibility?
random.seed(10)
np.random.seed(10)
torch.manual_seed(10)

test_path        = '/home/buosos/Documents/mritrans/Dirix/SegmentationTask/Segmentation_Task_PureSynthetic/test_data/unseen'
model_path       = '/home/buosos/Documents/Code/Segment4DFlow/'
out_path         = '/home/buosos/Documents/Code/Segment4DFlow/Results_inference/'

# For pureSyntetic training use ref_epoch = 19
# For mix use ref_epoch = 5
# For in vivo augmentation use ref_epoch = 10

if not os.path.exists(out_path):
    os.makedirs(out_path)
    
model_name = 'SegmentAorta'
ref_epoch   = 5
        
model = AttentionUNet(1,1,64,block='SingleConv',pool=True).to(torch.float).to(device)
checkpoint = torch.load(model_path + '/Model_2D_mix/'+model_name+'_%05d.pt' %ref_epoch, map_location = device)
model.load_state_dict(checkpoint['model_state_dict'])
# [patient,time,slice,X,Y]
mySigmoid   = torch.nn.Sigmoid()

patients = np.sort(next(os.walk(test_path))[2])

dice_scores = []
for patient_sel in patients[0:-1]:
    with h5py.File(test_path+'/'+patient_sel, 'r') as f:
        mask_sel  = f['mask'][:].astype(np.float32)
        image_sel = f['magnitude'][:].astype(np.float32)        
        for ii in range(image_sel.shape[0]):
            image_sel[ii,...] = normalize_and_standardize(image_sel[ii,...])

        nn_masks = np.zeros_like(mask_sel)

    with torch.no_grad():
        for i in range(image_sel.shape[0]):
            nn_masks[i,...] = getLargestCC(create_mask1ch(mySigmoid(model(torch.from_numpy(np.expand_dims(image_sel[i,...],1)).to(device)))).cpu().detach().numpy()[:,0,...])
    np.save(out_path+'/'+patient_sel+'_inference.npy',nn_masks)

    dice_values = []

    for i in range(mask_sel.shape[0]):
        dice_values.append(dice_score(nn_masks[i,...],mask_sel[i,...]))
    print(patient_sel,'dice mean/sdt %.2f(%.2f)' %(np.mean(dice_values),np.std(dice_values)))
    
    dice_scores.append(dice_values)
                               
np.save(out_path+'/DiceScores_invivo.npy',dice_values)

array = np.array(dice_scores)

# Flatten the array to a 1D NumPy array
flat_array = array.flatten()
print('Overall dice mean/sdt %.2f(%.2f)' %(np.mean(flat_array),np.std(flat_array)))

for i in range(len(dice_scores)):
    plt.plot(np.arange(len(dice_scores[i])),dice_scores[i],'o')
plt.savefig(out_path+'/DiceScores_invivo.png')





