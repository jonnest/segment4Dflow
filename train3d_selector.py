import torch
has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = 'mps' if has_mps else 'cuda' if has_gpu else 'cpu'
print("Running on ", device, '\n')
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
import sys


if device=='mps':
    sys.path.append('/Users/jonathan/Documents/code/segment4DFlow/medformer3D')
else:
    sys.path.append('/home/jnestmann/code/segment4DFlow/medformer3D')
    
from unet import UNet # as UNet_medformer
from attention_unet import AttentionUNet

from myFunctions.data_handling import *

# Reproducibility?
random.seed(10)
np.random.seed(10)
torch.manual_seed(10)

w_dir = '/Users/jonathan/Documents/applications/vesselSelectNet'

out_path = os.path.join(w_dir, 'output')
# img_path = os.path.join(w_dir, 'images')
img_path = '/Users/jonathan/Documents/data/deepFlow/volunteers'

model_dir = 'select_3D_n1'
model_name = 'SelectBrainVessels'

batch_numb = 12 # default 32
n_epochs   = 500
n_patch    = 32  # 64 
n_input    = 1
init_lr    = 1e-3

if not os.path.exists(out_path):
    os.makedirs(out_path)

if not os.path.isdir(out_path + '/' + model_dir + '/Results_3D'):
    os.makedirs(out_path + '/' + model_dir + "/Results_3D")

if not os.path.isdir(out_path + f'/{model_dir}/'):
    os.makedirs(out_path + f'/{model_dir}/')

t0 = time.time()

### 1 channel input
# masks, images = getAllPatches(img_path, NP=n_patch)
# train_dataset,val_dataset,test_dataset = split4Train(masks, images, split=0.8)

### 2 channel input
# TODO : rewrite patching, so that patches include whole domain
# TODO : check if patches take also edges and zero pad rest

# TODO : delete [:1] from patients
masks, images = getAllPatches(img_path, NP=n_patch, dims=3, mask_str="mask", image_str="network_seg")
# select number of patches to use
masks, images = masks[:2400], images[:2400]

train_dataset,val_dataset,test_dataset = split4Train(masks, images, split=0.8)

train_dataloader = DataLoader(train_dataset, batch_size=batch_numb, shuffle=True)#, num_workers=2)
val_dataloader   = DataLoader(val_dataset,   batch_size=batch_numb, shuffle=True)#, num_workers=2)
 
# In vivo testing
test_dataloader = DataLoader(test_dataset, batch_size=batch_numb, shuffle=True)#, num_workers=2)

### model : AttentionUNet(1,2,3,...)
# 1 : input channels
# 2 : number classes , (output channels?)
# 3 : base_ch
# model = UNet_medformer(1,1).to(torch.float).to(device)

### kernel size of 3d unet
kernel_size = [
    (3, 3, 3),  # For self.inc
    (3, 3, 3),  # For self.down1
    (3, 3, 3),  # For self.down2
    (3, 3, 3),  # For self.down3
    (3, 3, 3)   # For self.down4, self.up1, self.up2, self.up3, self.up4
]

# kernel_size = (3, 3, 3, 3, 3)
# scale = (2, 2, 2, 2)
scale = (1, 1, 1, 1)

# model = AttentionUNet(n_input,base_ch=16,scale=scale,kernel_size=kernel_size,block='SingleConv',pool=True).to(torch.float).to(device)
model = UNet(n_input,base_ch=16,scale=[2,2,2,2], kernel_size=[3,3,3,3,3], num_classes=1, block='ConvNormAct', pool=True, norm='bn')


def init_weights(m):
    if isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

model.apply(init_weights)

dice_loss  = torch.nn.BCEWithLogitsLoss() #(reduction='mean')
cross_loss = torch.nn.CrossEntropyLoss(reduction='mean')
mse_       = torch.nn.MSELoss(reduction='mean')

# optimizer
optimizer = optim.Adam(model.parameters(), lr = init_lr, betas = (0.5, 0.999))
# Choose a scheduler : StepLR, ReduceLROnPlateau, CosineAnnealingLR
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# Initialize ReduceLROnPlateau
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,              # The optimizer being used
#     mode='min',             # Mode can be 'min' or 'max'. Use 'min' for metrics like loss and 'max' for metrics like accuracy
#     factor=0.1,             # Factor by which the learning rate will be reduced (new_lr = lr * factor)
#     patience=10,            # Number of epochs with no improvement after which learning rate will be reduced
#     verbose=True,           # If True, prints a message to the console whenever the learning rate is reduced
#     threshold=0.0001,       # Threshold for measuring the new optimum, to only focus on significant changes
#     threshold_mode='rel',   # Mode can be 'rel' or 'abs'. 'rel' means the threshold is relative to the best observed value
#     cooldown=0,             # Number of epochs to wait before resuming normal operation after lr has been reduced
#     min_lr=0,               # A lower bound on the learning rate
#     eps=1e-8                # Minimal decay applied to lr; avoids it getting stuck at 0
# )

mySigmoid = torch.nn.Sigmoid()
loss_train, dice_train, dice_test = [], [], []
loss_val, dice_val   = [], []
learning_rate = []

for epoch_idx in range(n_epochs):

    model.train(True)
    batch_iter   = tqdm(train_dataloader, desc='train %i' %epoch_idx, leave=True)
    running_loss_t = 0
    running_loss_v = 0
    data_seen    = 0
    dice_t = 0
    for batch_idx, data_input in enumerate(batch_iter):
        
        optimizer.zero_grad()
        input_net = data_input['image'].to(device, dtype=torch.float)
        gt_label  = data_input['mask'].to(device, dtype=torch.float)
        nn_label  = mySigmoid(model(input_net))
        
        ### loss function
        loss      = mse_(gt_label, nn_label)
        # loss      = dice_loss(gt_label, nn_label)
        
        loss.backward()
        running_loss_t += loss.item()*input_net.shape[0]
        data_seen    += input_net.shape[0] 
        dice_t += dice_score(data_input['mask'].cpu().detach().numpy(), 
                             nn_label.cpu().detach().numpy()) * input_net.shape[0]
        batch_iter.set_postfix_str("train loss=%.4g, train dice =%.4g (%d)" % (running_loss_t/data_seen, dice_t/data_seen, data_seen))
        optimizer.step()
        
        
    # Save the current learning rate
    for param_group in optimizer.param_groups:
        learning_rate.append(param_group['lr'])
    
    loss_train.append(running_loss_t/data_seen)
    dice_train.append(dice_t/data_seen)
    
    data_seen_v = 0
    with torch.no_grad():
        batch_iter   = tqdm(val_dataloader, desc='val %i' %epoch_idx, leave=True)
        for batch_idx, data_input in enumerate(batch_iter):
            input_net = data_input['image'].to(device, dtype=torch.float)
            gt_label  = data_input['mask'].to(device, dtype=torch.float)
            nn_label  = create_mask1ch(mySigmoid(model(input_net))).cpu().detach().numpy()
            loss = dice_score(data_input['mask'].cpu().detach().numpy(), nn_label )
            running_loss_v += loss.item()*input_net.shape[0]
        
            data_seen_v    += input_net.shape[0] 
            
            batch_iter.set_postfix_str("train loss=%.4g, val_dice=%.4g (%d)" % (running_loss_t/data_seen, running_loss_v/data_seen_v, data_seen_v))

    # Step the schedulerm, update the learning rate // with the validation loss
    scheduler.step()
    
    loss_val.append(running_loss_v/data_seen_v)
    dice_val.append(running_loss_v/data_seen_v)
    
    plt.plot(learning_rate,label='lr')
    # plt.plot(loss_val,label='val')
    plt.xlabel('epochs')
    plt.ylabel('lr')
    plt.legend()
    plt.savefig(out_path+f'/{model_dir}/learning_rate.png', dpi=200)
    plt.close()
    
    plt.plot(dice_train, label='train')
    plt.plot(dice_val, label='val')
    plt.xlabel('epochs')
    plt.ylabel('dice score')
    plt.legend()
    plt.savefig(out_path+f'/{model_dir}/dice_score.png', dpi=200)
    plt.close()

    plt.plot(loss_train,label='train')
    # plt.plot(loss_val,label='val')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(out_path+f'/{model_dir}/loss.png', dpi=200)
    plt.close()
    
    np.save(out_path+f'/{model_dir}/train_loss.npy',    loss_train)
    np.save(out_path+f'/{model_dir}/val_loss.npy',      loss_val)
    np.save(out_path+f'/{model_dir}/train_dice.npy',    dice_train)
    np.save(out_path+f'/{model_dir}/val_dice.npy',      dice_val)
    np.save(out_path+f'/{model_dir}/learning_rate.npy', learning_rate)
    
    # TODO : does not work in 3d yet
    
    # if epoch_idx%10 == 0:
    #     with torch.no_grad():
    #         pick_slices = np.random.choice(len(val_dataset), 15)
    #         mask_test   = val_dataset[pick_slices]['mask']
    #         images_test = val_dataset[pick_slices]['image']
    #         nn_label    = create_mask1ch(mySigmoid(model(torch.from_numpy(images_test).float().to(device)))).cpu().detach().numpy()

    #         img1  = np.concatenate((images_test[0,0,...,0], images_test[1,0,...,0],images_test[2,0,...,0], images_test[3,0,...,0]),1)
    #         img2  = np.concatenate((images_test[0,1,...,0], images_test[1,1,...,0],images_test[2,1,...,0], images_test[3,1,...,0]),1)

    #         mask = np.concatenate((nn_label[0,0,...],    nn_label[1,0,...],   nn_label[2,0,...],    nn_label[3,0,...]),1)
    #         gt   = np.concatenate((mask_test[0,0,...],   mask_test[1,0,...],  mask_test[2,0,...],   mask_test[3,0,...]),1)

    #         masked_data = np.ma.masked_where(mask == 0, mask)
    #         masked_gt   = np.ma.masked_where(gt == 0, gt)
            
    #         plt.imshow(img, cmap='gray')
    #         plt.imshow(masked_data, cmap=ListedColormap(['blue']), alpha=0.2, interpolation='none')
    #         plt.imshow(masked_gt,   cmap=ListedColormap(['red']), alpha=0.2, interpolation='none')
    #         plt.savefig(out_path + '/' + model_dir + '/Results_3D/epoch_%06d.png' %(epoch_idx), bbox_inches = 'tight',dpi=300)
    #         plt.close()
    
    # save model every i-th epoch
    ith_epoch = 10
    if epoch_idx%ith_epoch == 0:
        torch.save({
                   'epoch'                         : epoch_idx,
                   'model_state_dict'              : model.state_dict(),
                   'optimizer_state_dict'          : optimizer.state_dict()
                    }, out_path +  f'/{model_dir}/'+model_name+'_%05d.pt' %epoch_idx) 


    ### test the model every i-th epoch
    ith_epoch = 10
    if epoch_idx%ith_epoch == 0:
        loss_t = 0 
        data_seen_t = 0
        with torch.no_grad():
            batch_iter   = tqdm(test_dataloader, desc='test %i' %epoch_idx, leave=True)
            for batch_idx, data_input in enumerate(batch_iter):
                optimizer.zero_grad()
                input_net = data_input['image'].to(device, dtype=torch.float)
                gt_label  = data_input['mask'].to(device, dtype=torch.float)
                nn_label  = create_mask1ch(mySigmoid(model(input_net))).cpu().detach().numpy()
                loss = dice_score(data_input['mask'].cpu().detach().numpy(),nn_label)
                loss_t += loss.item()*input_net.shape[0]

                data_seen_t    += input_net.shape[0] 
                batch_iter.set_postfix_str("test dice=%.4g (%d)" % (loss_t/data_seen_t, data_seen_t))
        
        print('Test dice is ',loss_t/data_seen_t) 
        dice_test.append(loss_t/data_seen_t)
        n_iter = len(dice_test)
        plt.plot(ith_epoch*np.linspace(1,n_iter,n_iter), dice_test, label='test')
        plt.xlabel('epochs')
        plt.ylabel('dice score')
        plt.legend()
        plt.savefig(out_path+f'/{model_dir}/test_dice.png', dpi=200)
        plt.close()

        # TODO : does not work in 3d yet

        # pick_slices = np.random.choice(len(val_dataset), 15)
        # mask_test   = val_dataset[pick_slices]['mask']
        # images_test = val_dataset[pick_slices]['image']
        # nn_label    = create_mask1ch(mySigmoid(model(torch.from_numpy(images_test).float().to(device)))).cpu().detach().numpy()

        # for ii in range(15):
        #     img  = images_test[ii,0,...]
        #     mask = nn_label[ii,0,...]
        #     gt   = mask_test[ii,0,...]

        #     masked_data = np.ma.masked_where(mask == 0, mask)
        #     masked_gt   = np.ma.masked_where(gt == 0, gt)
        #     plt.imshow(img, cmap='gray')
        #     plt.imshow(masked_data, cmap=ListedColormap(['blue']), alpha=0.2, interpolation='none')
        #     plt.imshow(masked_gt,   cmap=ListedColormap(['red']), alpha=0.2, interpolation='none')
        #     plt.savefig(out_path + '/' + model_dir + '/Results_3D/invivo_%04d.png' %(ii), bbox_inches = 'tight',dpi=300)
        #     plt.close()

t1 = time.time()
print("\n--- Totat time needed ---")
print(round(t1-t0,2), " seconds")