import torch
has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = 'mps' if has_mps else 'cuda' if has_gpu else 'cpu'
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
import sys

if device=='mps':
    sys.path.append('/Users/jonathan/Documents/code/segment4DFlow/medformer3D')
else:
    sys.path.append('/home/jnestmann/code/segment4DFlow/medformer3D')
    
from unet import UNet # as UNet_medformer
from attention_unet import AttentionUNet
from dualpath_unet import DualPathUNet

from myFunctions.data_handling import *

# Reproducibility?
random.seed(10)
np.random.seed(10)
torch.manual_seed(10)

if device=='mps':
    w_dir = '/Users/jonathan/Documents/applications/unet'
    # img_path = '/Users/jonathan/Documents/data/deepFlow/volunteers'
    img_path = os.path.join(w_dir, 'images')
elif device=='cuda':
    # data_dir = '/mnt/itetstor/jnestmann/mritrans_ibtscratch/nestmann/unet'
    data_dir = '/mnt/itetstor/jnestmann/digitaltwin_ibtscratch/nestmann/unet'
    w_dir = '/home/jnestmann/applications/unet'
    img_path = os.path.join(data_dir, 'images_volunteer')

out_path = os.path.join(w_dir, 'output')

batch_numb      = 12 # default 32
n_epochs        = 1000
n_patch         = 64  # 64 
n_input         = 2
init_lr         = 1e-3
nr_base_filters = 32 # 32
scheduler       = 'cosine' 

# TODO : augmentation

model_dir = f'Model_aug_bl4_vol5_b{batch_numb}_p{n_patch}_f{nr_base_filters}_lr{scheduler}'
model_name = 'SegmentBrainVessels'

parameters = {
    'batch_numb' : batch_numb,
    'n_epochs' : n_epochs,
    'n_patch' : n_patch,
    'n_input' : n_input,
    'init_lr' : init_lr,
    'nr_base_filters' : nr_base_filters,
    'scheduler' : scheduler,
}

if not os.path.exists(out_path):
    os.makedirs(out_path)

if not os.path.isdir(out_path + f"/{model_dir}"):
    os.makedirs(out_path + f"/{model_dir}")
    
if not os.path.isdir(out_path + "/" + model_dir + "/Results_3D"):
    os.makedirs(out_path + "/" + model_dir + "/Results_3D")
    
write_dict_to_file(parameters, out_path+f"/{model_dir}/parameters.txt")

t0 = time.time()

### 1 channel input
# masks, images = getAllPatches(img_path, NP=n_patch)
# train_dataset,val_dataset,test_dataset = split4Train(masks, images, split=0.8)

### 2 channel input
# TODO : rewrite patching, so that patches include whole domain
# TODO : check if patches take also edges and zero pad rest
# TODO : delete [:1] from patients

# masks, images1, images2 = getAllPatches2Channels(img_path, channel1='magnitude', channel2='cd', NP=n_patch)
masks, images = getPatches(img_path, channels=['magnitude','cd'], NP=64, dims=3)

print("\n\nmask shape      ", masks.shape)
print("images shape    ", images.shape)

masks, images = augment_and_expand_3d_patches(masks, images, num_aug=4, rotation=True, flip=True)

print("\n\nmask shape      ", masks.shape)
print("images shape    ", images.shape)

# nr_cut = 1000 # 2400
# masks, images1, images2 = masks[:nr_cut], images1[:nr_cut], images2[:nr_cut]

'''
train_dataset,val_dataset,test_dataset = split4TrainAll(masks, images, split=0.8)
train_dataloader = DataLoader(train_dataset, batch_size=batch_numb, shuffle=True)#, num_workers=2)
val_dataloader   = DataLoader(val_dataset,   batch_size=batch_numb, shuffle=True)#, num_workers=2)
'''

masks, local_images, global_images = getPatchesWithGlobalContext(img_path, channels=['magnitude','cd'], NP=64, dims=3)
train_dataset, val_dataset, test_dataset = split4TrainAll(masks, local_images, global_images, split=0.8)
train_dataloader = DataLoader(train_dataset, batch_size=batch_numb, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_numb, shuffle=True)


# In vivo testing
test_dataloader = DataLoader(test_dataset, batch_size=batch_numb, shuffle=True)#, num_workers=2)

### model : AttentionUNet(1,2,3,...)
# 1 : input channels
# 2 : number classes , (output channels?)
# 3 : base_ch
# model = UNet_medformer(1,1).to(torch.float).to(device)

# kernel size of 3d unet
kernel_size= [3,3,3,3,3]
scale = [2,2,2,2]

# Assuming you have a local UNet and a global UNet or a simple CNN
local_model = UNet(n_input, base_ch=nr_base_filters, scale=[2,2,2,2], kernel_size=kernel_size, num_classes=1, block='SingleConv',pool=True).to(torch.float).to(device)
# or any other global context model
global_model = UNet(n_input, base_ch=nr_base_filters, scale=[4,4,4,4], kernel_size=kernel_size, num_classes=1, block='SingleConv',pool=True).to(torch.float).to(device)

model = DualPathUNet(local_model, global_model, num_classes=1).to(device)

'''
if device=='mps':
    model = UNet(n_input,base_ch=nr_base_filters,scale=[2,2,2,2], kernel_size=kernel_size, num_classes=1, block='ConvNormAct', pool=True, norm='bn')
elif device=='cuda':
    # model = AttentionUNet(n_input,base_ch=64,scale=scale,kernel_size=kernel_size,block='SingleConv',pool=True).to(torch.float).to(device)
    model = UNet(n_input,base_ch=nr_base_filters,scale=[2,2,2,2], kernel_size=kernel_size, num_classes=1, block='ConvNormAct', pool=True, norm='bn').to(torch.float).to(device)
'''

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
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

mySigmoid = torch.nn.Sigmoid()
loss_train, dice_train, dice_test = [], [], []
loss_val, dice_val   = [], []
learning_rate = []


# TODO : introduce in main loop
'''
for epoch_idx in range(n_epochs):
    model.train(True)
    batch_iter = tqdm(train_dataloader, desc='train %i' %epoch_idx, leave=True)
    running_loss_t = 0
    data_seen = 0
    for batch_idx, data_input in enumerate(batch_iter):
        optimizer.zero_grad()
        local_input = data_input['local_image'].to(device, dtype=torch.float32)
        global_input = data_input['global_image'].to(device, dtype=torch.float32)
        gt_label = data_input['mask'].to(device, dtype=torch.float32)
        
        nn_label = model(local_input, global_input)
        loss = mse_(gt_label, nn_label)
        loss.backward()
        optimizer.step()
        
        running_loss_t += loss.item() * local_input.shape[0]
        data_seen += local_input.shape[0]
        batch_iter.set_postfix_str("train loss=%.4g (%d)" % (running_loss_t/data_seen, data_seen))
        
    # Continue with validation, learning rate scheduling, etc.


'''
for epoch_idx in range(n_epochs):

    model.train(True)
    batch_iter   = tqdm(train_dataloader, desc='train %i' %epoch_idx, leave=True)
    running_loss_t = 0
    running_loss_v = 0
    data_seen    = 0
    dice_t = 0
    for batch_idx, data_input in enumerate(batch_iter):
        optimizer.zero_grad()
        input_net = data_input['image'].to(device, dtype=torch.float32) # dtype=torch.float)
        gt_label  = data_input['mask'].to(device, dtype=torch.float32)  # dtype=torch.float)

        if device=='mps':
            nn_label  = mySigmoid(model(input_net))
        elif device=='cuda':
            nn_label  = mySigmoid(model(input_net.to(device, dtype=torch.float32)))
        
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

    # validation loss per iteration
    loss_val_it = 1-running_loss_v/data_seen_v
    # Step the schedulerm, update the learning rate // with the validation loss
    scheduler.step() 
    # scheduler.step(loss_val_it) # for ReduceLROnPlateau
    
    loss_val.append(loss_val_it)
    dice_val.append(running_loss_v/data_seen_v)
    
    plt.plot(learning_rate,label='lr')
    # plt.plot(loss_val,label='val')
    plt.xlabel('epochs')
    plt.ylabel('lr')
    plt.yscale('log')
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