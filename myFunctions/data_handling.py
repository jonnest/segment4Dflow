import torch
# has_gpu = torch.cuda.is_available()
# has_mps = torch.backends.mps.is_built()
# device = 'mps' if has_mps else 'cuda' if has_gpu else 'cpu'
# print("Running on ", device)

# torch.set_default_dtype(torch.float32)
# torch.set_default_dtype(torch.float16)
from torch.utils.data import Dataset
import torchvision
from skimage.transform import resize
import h5py
import os
import numpy as np

from torchvision import transforms as T
from PIL import Image
import pyvista as pv

import matplotlib.pyplot as plt
from scipy import ndimage
import skimage as ski
from skimage import io, img_as_float
import random


def getPatchesWithGlobalContext(img_path, channels, NP=64, dims=3):
    # Assuming the getPatches function returns patches and full images
    patches, full_images = getPatches(img_path, channels, NP=NP, dims=dims)
    
    return patches, full_images


# Write dictionary to a text file
def write_dict_to_file(dictionary, file_path):
    with open(file_path, 'w') as file:
        for key, value in dictionary.items():
            file.write(f"{key}: {value}\n")
            
            
def getMultipleLargestCC(segmentation, nr_masks=1):
    # denoise the image with a Gaussian filter
    # segmentation = ski.filters.gaussian(segmentation, sigma=0.2)
    
    # CC : connected component
    labels = label(segmentation)
    assert(labels.max() != 0)  # Assume at least 1 connected component
    
    component_sizes = np.bincount(labels.flat)# [1:]
    print("component sizes ", component_sizes)
    
    largest_labels = np.argsort(component_sizes)[-nr_masks:][::-1] + 1  # +1 to adjust for background exclusion
    print("largest_labels ", largest_labels)
    # largestCCs = [(labels == label) for label in largest_labels]
    # Create a binary mask containing all four largest components
    largestCCs_mask = np.isin(labels, largest_labels)
    
    return largestCCs_mask


def getConnectivity(segmentation, nr_masks=1):
    # pv_mask = np.reshape(segmentation, -1, order='F')
    grid = pv.ImageData()
    grid.dimensions = [x_ for x_ in segmentation.shape]
    grid['data'] = np.reshape(np.round(segmentation).astype('double'),-1) #,order='F')
    
    all_regions = grid.connectivity('all')
    region_ids = np.unique(all_regions['RegionId'])
    select_region_ids = region_ids[:nr_masks]  # All region ids except '0'
    pv_segmentation_mask = grid.connectivity('specified', select_region_ids)

    segmentation_mask = np.reshape(pv_segmentation_mask['data'],segmentation.shape,order='F')
    # segmentation_mask = np.reshape(pv_segmentation_mask, segmentation.shape, order='F')
    
    return segmentation_mask


def getLargestSegmentations(segmentation, nr_masks=1):
    ## connectivity
    labels, num_labels = ndimage.label(segmentation)
    # Calculate sizes of each labeled region
    label_sizes = ndimage.sum(segmentation, labels, range(num_labels + 1))
    largest_labels = np.argsort(label_sizes)[-nr_masks:]
    print("Largest labels  ", largest_labels)
    
    # Apply the mask to the original segmentation
    segmentation_mask = np.isin(labels, largest_labels)
    
    return segmentation_mask

def getLargestCC(segmentation):
    # denoise the image with a Gaussian filter
    segmentation = ski.filters.gaussian(segmentation, sigma=0.2)
    
    labels = label(segmentation, connectivity=1)
    unique, counts = np.unique(labels, return_num=True)
    list_seg=list(zip(unique, counts))[1:] # the 0 label is by default background so take the rest
    largest=max(list_seg, key=lambda x:x[1])[0]
    labels_max=(labels == largest).astype(int)

    return labels_max


def largest_connected_component(segmentation_mask):
    # denoise the image with a Gaussian filter
    segmentation = ski.filters.gaussian(segmentation_mask, sigma=0.2)
    
    # Label connected components
    labeled_array, num_features = label(segmentation_mask, return_num=True)
    
    # If there are no features, return the original mask
    if num_features == 0:
        return segmentation_mask
    
    # Find the sizes of the components
    sizes = np.bincount(labeled_array.ravel())
    
    # We start from 1 because 0 is the background
    largest_label = sizes[1:].argmax() + 1
    
    # Create a mask of the largest component
    largest_component_mask = labeled_array == largest_label
    
    return largest_component_mask


def getLargestCC_org(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC


def largestCC_pv(segmentation_mask):
    # Convert the 3D segmentation mask to a PyVista mesh
    grid = pv.ImageData()
    grid.dimensions = np.array(segmentation_mask.shape) + 1
    grid.spacing = (1, 1, 1)
    grid.origin = (0, 0, 0)
    
    # Add the binary segmentation mask as a scalar field
    grid.cell_data['data'] = segmentation_mask.flatten(order='F')
    
    # Apply the connectivity filter
    connectivity_filter = grid.connectivity()
    
    # Extract the largest connected region
    largest_component = connectivity_filter.threshold(0.5, invert=True, scalars="RegionId")
    
    # Convert back to a NumPy array
    largest_component_mask = np.zeros_like(segmentation_mask)
    largest_component_mask[largest_component.point_data['RegionId'].astype(int) - 1] = 1
    
    return largest_component_mask



def create_mask1ch(probabilities, threshold=0.5):
    mask = torch.zeros_like(probabilities, dtype=torch.float)  # Initialize mask with zeros
    mask[probabilities >= threshold] = 1  # Set values above or equal to threshold to 1
    return mask

def dice_score( inputs, targets, smooth=1e-6):        
    inputs = inputs.flatten()
    targets = targets.flatten()
    
    intersection = (inputs * targets).sum()                            
    dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
    
    return dice

def normalize_and_standardize(image):
    # normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
    # return (normalized - np.mean(normalized)) / np.std(normalized)
    normalized = (image - np.min(image)) 
    return (normalized) / np.max(normalized)


class data2D(Dataset):

    def __init__(self, masks4D, images4D):
        
        # self.masks  = np.expand_dims(masks4D,1)
        self.masks  = masks4D
        # print("data2D : Mask shape ", self.masks.shape)
        # self.images = np.expand_dims(images4D,1)
        self.images  = images4D
        self.n = self.masks.shape[0]
        
    def __getitem__(self, index):

        return {'mask':  self.masks[index,...],
                'image': self.images[index,...]}

    def __len__(self):
        return self.n


def list_directories(directory):
    # List all entries in the given directory
    entries = os.listdir(directory)
    
    # Filter out only directories
    directories = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
    
    return directories


def getAllPatches(img_path, NP=64, dims=2, mask_str="mask", image_str="cd"):
    ''' Get stack of patches from all available data sets
    '''
    # walk through patients
    patients = list_directories(img_path)

    start_add = True
    # Training dataset
    for patient_sel in patients:
        print(patient_sel)
        
        # mask name depending on folder structure
        # mask = pv.read(os.path.join(img_path,patient_sel,'segmentation_'+patient_sel+'.vtk'))
        mask = pv.read(os.path.join(img_path,patient_sel,f'{mask_str}_.vtk'))
        
        # image = pv.read(os.path.join(img_path,patient_sel,f'{image_str}_'+patient_sel+'.vtk'))
        image = pv.read(os.path.join(img_path,patient_sel,f'{image_str}_.vtk'))
                        
        # choose string depending on segmenation variant
        try:
            mask = np.reshape(mask['scalars'], mask.dimensions, order='F')
        except:
            mask = np.reshape(mask['data'], mask.dimensions, order='F')
        image = np.reshape(image['data'], image.dimensions, order='F')    
            
        # patching
        if dims == 2:
            for ii in range(image.shape[-1]):
                image[...,ii] = normalize_and_standardize(image[...,ii])      
            mask_p, image_p = patching2d(mask, image, NP)
            
        elif dims == 3:
            mask_p, image_p = patching3d(mask, image, NP)

        # store all patients in mask
        if start_add:
            masks_p  = mask_p
            images_p = image_p
            start_add = False
        else:
            masks_p  = np.concatenate((masks_p,mask_p),0)
            images_p = np.concatenate((images_p,image_p),0)

    print("\nShape of valid masks : ", masks_p.shape, "\n")
        
    return masks_p, images_p
        
        
        
def getData(img_path, channels=['magnitude','cd'], dims=3, patching=True, NP=64):
    ''' Get stack of patches from all available data sets
    '''
    # walk through patients
    patients = list_directories(img_path)

    print(patients)
    start_add = True
    # Training dataset
    for patient_sel in patients:
        print("\n--- ", patient_sel, " ---")
        
        try:
            mask = pv.read(os.path.join(img_path,patient_sel,'segmentation_'+patient_sel+'.vtk'))
            print(f"For {patient_sel} used segmentation_{patient_sel}.vtk version")
        except:
            mask = pv.read(os.path.join(img_path,patient_sel,'mask_.vtk'))
            print(f"For {patient_sel} used mask_.vtk version")
        
        for i in range(len(channels)):
            if i==0:
                image = pv.read(os.path.join(img_path,patient_sel, f'{channels[0]}_'+patient_sel+'.vtk'))
                image = np.reshape(image['data'], image.dimensions, order='F')
                image = np.expand_dims(image, 0)
            else:
                image_i = pv.read(os.path.join(img_path,patient_sel, f'{channels[i]}_'+patient_sel+'.vtk'))
                image_i = np.reshape(image_i['data'], image_i.dimensions, order='F')
                image_i = np.expand_dims(image_i, 0)
                image  = np.concatenate((image,image_i),0)
        
        print("shape of image after reading and concatentation ", image.shape)
        # choose string depending on segmenation variant
        try:
            mask = np.reshape(mask['scalars'], mask.dimensions, order='F')
        except:
            mask = np.reshape(mask['data'], mask.dimensions, order='F')
        
        mask = np.expand_dims(mask, 0)
        
        print("\nmask shape after reading : ", mask.shape, "\n")
        
        # patching
        if patching:
            # TODO : rewrite more GENERAL
            if dims == 2:
                for i in range(len(channels)):
                    image[:,i,...] = normalize_and_standardize(image)[:,i,...]   
                mask, image = patching2d(mask, image, NP)
                
            elif dims == 3:
                for i in range(len(channels)):
                    image[:,i,...] = normalize_and_standardize(image)[:,i,...]        
                mask, image = patching3d(mask, image, NP)
        else:
            # reshape array for z-dir as first dimension, slicing in z-dir
            mask = np.transpose(mask, (3, 0, 1, 2))
            image = np.transpose(image, (3, 0, 1, 2))
            
        print("Shape mask ", mask.shape)
        
        # store all patients in mask
        if start_add:
            masks  = mask
            images = image
            start_add = False
        else:
            masks  = np.concatenate((masks,mask),0)
            images = np.concatenate((images,image),0)

    print("\nShape of valid masks : ", masks.shape, "\n")
        
    return masks, images
        

        
def split4Train(masks, images, split=0.8):
    ''' Split all the available patches into training, validation and testing sets.
    The arrays masks and images have the same size
    '''

    # calculate split indices
    idx1 = int(masks.shape[0]*split)
    idx2 = int(masks.shape[0]*(split + (1-0.8)/2))

    train_masks = masks[:idx1]
    valid_masks = masks[idx1:idx2]
    test_masks = masks[idx2:]

    train_images = images[:idx1]
    valid_images = images[idx1:idx2]
    test_images = images[idx2:]
    
    train_dataset = data2D(train_masks,train_images)
    valid_dataset = data2D(valid_masks,valid_images)
    test_dataset = data2D(test_masks,test_images)

    return train_dataset, valid_dataset, test_dataset


def split4Train2Channels(masks, images1, images2, split=0.8):
    ''' Split all the available patches into training, validation and testing sets.
    The arrays masks and images have the same size
    '''

    # calculate split indices
    idx1 = int(masks.shape[0]*split)
    idx2 = int(masks.shape[0]*(split + (1-0.8)/2))

    train_masks = masks[:idx1]
    valid_masks = masks[idx1:idx2]
    test_masks = masks[idx2:]

    train_images1 = images1[:idx1]
    valid_images1 = images1[idx1:idx2]
    test_images1 = images1[idx2:]
    
    train_images2 = images2[:idx1]
    valid_images2 = images2[idx1:idx2]
    test_images2 = images2[idx2:]
    
    train_images= np.stack((train_images1, train_images2), axis=1)
    valid_images = np.stack((valid_images1, valid_images2), axis=1)
    test_images = np.stack((test_images1, test_images2), axis=1)
    
    print("\nshape train_images  ", train_images.shape)
    print("train_masks         ", train_masks.shape)
    
    train_dataset = data2D(train_masks,train_images)
    valid_dataset = data2D(valid_masks,valid_images)
    test_dataset = data2D(test_masks,test_images)

    return train_dataset, valid_dataset, test_dataset


def split4TrainAll(masks, images, split=0.8):
    ''' Split all the available patches into training, validation and testing sets.
    The arrays masks and images have the same size
    '''

    # calculate split indices
    idx1 = int(masks.shape[0]*split)
    idx2 = int(masks.shape[0]*(split + (1-0.8)/2))

    train_masks = masks[:idx1]
    valid_masks = masks[idx1:idx2]
    test_masks = masks[idx2:]

    train_images = images[:idx1]
    valid_images = images[idx1:idx2]
    test_images = images[idx2:]
    
    print("\nshape train_images  ", train_images.shape)
    print("train_masks         ", train_masks.shape)
    
    train_dataset = data2D(train_masks,train_images)
    valid_dataset = data2D(valid_masks,valid_images)
    test_dataset = data2D(test_masks,test_images)

    return train_dataset, valid_dataset, test_dataset

'''
def patching2d(mask, image, NP):
    
    Nx, Ny, Nz = mask.shape
    overlap = NP // 2
    stride = NP - overlap
    
    # Calculate the number of patches in x and y directions
    num_patches_x = (Nx - overlap) // stride
    num_patches_y = (Ny - overlap) // stride
    
    # Calculate the total number of patches
    M = num_patches_x * num_patches_y
    
    # Initialize the array to store patches
    mask_p = np.zeros((M*Nz, NP, NP))
    image_p = np.zeros((M*Nz, NP, NP))
    
    patch_index = 0
    for z in range(Nz):
        for i in range(0, Nx - NP + 1, stride):
            for j in range(0, Ny - NP + 1, stride):
                # Extract the patch
                # print("x  ", i, " -  ", i+NP)
                # print("y  ", j, " -  ", j+NP)
                mask_p[patch_index] = mask[i:i+NP, j:j+NP, z]
                image_p[patch_index] = image[i:i+NP, j:j+NP, z]
                patch_index += 1
    
    # Remove the all-zero slices
    non_zero_slices = ~np.all(mask_p == 0, axis=(1, 2))
    mask_p = mask_p[non_zero_slices]
    # use same non-zero slices for image
    image_p = image_p[non_zero_slices]
    
    # Generate a random permutation of row indices
    permutation = np.random.permutation(mask_p.shape[0])

    # Shuffle both arrays using the same permutation
    mask_p = mask_p[permutation]
    image_p = image_p[permutation]
    
    return mask_p, image_p
'''


'''
def patching3d(mask, image, NP):
    
    Nx, Ny, Nz = mask.shape[:3]
    overlap = NP // 2
    stride = NP - overlap
    
    # Calculate the number of patches in x, y, and z directions
    num_patches_x = (Nx - overlap) // stride
    num_patches_y = (Ny - overlap) // stride
    num_patches_z = (Nz - overlap) // stride
    
    # Calculate the total number of patches
    M = num_patches_x * num_patches_y * num_patches_z
    
    # Initialize the array to store patches
    mask_p = np.zeros((M, NP, NP, NP))
    image_p = np.zeros((M, NP, NP, NP))
    
    patch_index = 0
    for i in range(0, Nx - NP + 1, stride):
        for j in range(0, Ny - NP + 1, stride):
            for k in range(0, Nz - NP + 1, stride):
                # Extract the patch
                mask_p[patch_index] = mask[i:i+NP, j:j+NP, k:k+NP]
                image_p[patch_index] = image[i:i+NP, j:j+NP, k:k+NP]
                patch_index += 1
    
    # Remove the all-zero patches
    non_zero_patches = ~np.all(mask_p == 0, axis=(1, 2, 3))
    mask_p = mask_p[non_zero_patches]
    image_p = image_p[non_zero_patches]
    
    # Generate a random permutation of row indices
    permutation = np.random.permutation(mask_p.shape[0])

    # Shuffle both arrays using the same permutation
    mask_p = mask_p[permutation]
    image_p = image_p[permutation]
    
    return mask_p, image_p
'''

def patching3d(mask, image, NP):
    ''' 
    - image (numpy array): Array of shape (C, NX, NY, NZ), where C is 1 or 2 (input channels).
    '''
    # number of channels
    C = image.shape[0]
    
    Nx, Ny, Nz = mask.shape[-3:]
    overlap = NP // 2
    stride = NP - overlap
    
    # Calculate the number of patches in x, y, and z directions
    num_patches_x = (Nx - overlap) // stride
    num_patches_y = (Ny - overlap) // stride
    num_patches_z = (Nz - overlap) // stride
    
    # Calculate the total number of patches
    M = num_patches_x * num_patches_y * num_patches_z

    # Initialize the array to store patches
    mask_p = np.zeros((M, 1, NP, NP, NP))
    image_p = np.zeros((M, C, NP, NP, NP))

    patch_index = 0
    for i in range(0, Nx - NP + 1, stride):
        for j in range(0, Ny - NP + 1, stride):
            for k in range(0, Nz - NP + 1, stride):
                # Extract the patch
                # TODO : check if mask[...] is correct, before mask[:,]
                mask_p[patch_index] = mask[..., i:i+NP, j:j+NP, k:k+NP]
                image_p[patch_index] = image[..., i:i+NP, j:j+NP, k:k+NP]
                patch_index += 1
    
    # Remove the all-zero patches
    non_zero_patches = ~np.all(mask_p[:,0,...] == 0, axis=(-3, -2, -1))
    mask_p = mask_p[non_zero_patches]
    # use same non-zero slices for image
    image_p = image_p[non_zero_patches]

    # Generate a random permutation of row indices
    permutation = np.random.permutation(mask_p.shape[0])

    # Shuffle both arrays using the same permutation
    mask_p = mask_p[permutation]
    image_p = image_p[permutation]
    
    return mask_p, image_p



def patching2d(mask, image, NP):
    ''' 
    - image (numpy array): Array of shape (C, NX, NY, NZ), where C is 1 or 2 (input channels).
    '''
    # number of channels
    C = image.shape[0]
    
    Nx, Ny, Nz = mask.shape
    overlap = NP // 2
    stride = NP - overlap
    
    # Calculate the number of patches in x and y directions
    num_patches_x = (Nx - overlap) // stride
    num_patches_y = (Ny - overlap) // stride
    
    # Calculate the total number of patches
    M = num_patches_x * num_patches_y
    
    # Initialize the array to store patches
    mask_p = np.zeros((M*Nz, NP, NP))
    image_p = np.zeros((M*Nz, C, NP, NP))
    
    patch_index = 0
    for z in range(Nz):
        for i in range(0, Nx - NP + 1, stride):
            for j in range(0, Ny - NP + 1, stride):
                # Extract the patch
                mask_p[patch_index] = mask[..., i:i+NP, j:j+NP, z]
                image_p[patch_index] = image[..., i:i+NP, j:j+NP, z]
                patch_index += 1
    
    # Remove the all-zero slices
    non_zero_slices = ~np.all(mask_p == 0, axis=(1, 2))
    mask_p = mask_p[non_zero_slices]
    # use same non-zero slices for image
    image_p = image_p[non_zero_slices]
    
    # Generate a random permutation of row indices
    permutation = np.random.permutation(mask_p.shape[0])

    # Shuffle both arrays using the same permutation
    mask_p = mask_p[permutation]
    image_p = image_p[permutation]
    
    return mask_p, image_p, image_p


def infer_3d_unet(model, volume, patch_size, overlap, mySigmoid):
    """
    Perform patch-wise inference on a large 3D volume with 2 input channels and 1 output channel.
    
    :param model: The 3D U-Net model.
    :param volume: The large 3D volume to segment (PyTorch tensor with 2 channels).
    :param patch_size: The size of the patches (tuple of 3 integers).
    :param overlap: The overlap between patches (tuple of 3 integers).
    
    :return: The segmented 3D volume (single output channel).
    """
    if isinstance(volume, np.ndarray):
        volume = torch.tensor(volume)
    
    _, D, H, W = volume.shape  # Assuming volume is of shape [2, D, H, W]
    segmented_volume = torch.zeros((1, D, H, W), dtype=volume.dtype)  # Output has 1 channel
    # weight map keeps track of how many times a voxel is predicted
    weight_map = torch.zeros((1, D, H, W), dtype=volume.dtype)
    
    for x in range(0, D, patch_size[0] - overlap[0]):
        for y in range(0, H, patch_size[1] - overlap[1]):
            for z in range(0, W, patch_size[2] - overlap[2]):
                # Define the start and end of the patch
                x_start, y_start, z_start = x, y, z
                print("\nx,y,z start  ", x,y,z)

                x_end = min(x_start + patch_size[0], D)
                y_end = min(y_start + patch_size[1], H)                
                z_end = min(z_start + patch_size[2], W)

                # Extract the patch
                patch = volume[:, x_start:x_end , y_start:y_end, z_start:z_end]
                print("Shape of patch  ", patch.shape)
                
                # Add padding if necessary
                # torch padding, pad from last dimension
                padded_patch = torch.nn.functional.pad(patch, (0, patch_size[2] - patch.shape[3],   # z-dir
                                                               0, patch_size[1] - patch.shape[2],   # y-dir
                                                               0, patch_size[0] - patch.shape[1]))  # x-dir
                
                # Convert the input tensor to float32 before passing to the model
                padded_patch = padded_patch.float()
                padded_patch = np.expand_dims(padded_patch,0)
                print("Shape of padded patch  ", padded_patch.shape)
                
                # speed up inference by decreasing precision (autocast) and turning of gradient computation
                with torch.no_grad():
                    # with autocast():
                    # pred_patch = model(padded_patch.unsqueeze(0)).squeeze(0)  # Model expects input shape [batch_size, channels, D, H, W]
                    pred_patch = create_mask1ch(mySigmoid(model(torch.from_numpy(padded_patch).float().to(device)))).cpu().detach().numpy()
                    # pred_patch = create_mask1ch(mySigmoid(model(padded_patch.to(device)))).cpu().detach().numpy()
                    
                # print("After inference, pred_patch shape  ", pred_patch.shape)
                # Remove padding from prediction if necessary
                pred_patch = pred_patch[..., :patch.shape[-3], :patch.shape[-2], :patch.shape[-1]]
                # print("After removing padding, pred_patch shape  ", pred_patch.shape)
                
                # Accumulate results (assuming output is 1 channel)
                segmented_volume[..., x_start:x_end, y_start:y_end, z_start:z_end] += pred_patch
                weight_map[:, x_start:x_end, y_start:y_end,z_start:z_end] += 1

                
    # Normalize the segmented volume
    segmented_volume /= weight_map
    return segmented_volume


def augment_and_expand_3d_patches(mask_patches, input_patches, num_aug=3, 
                                  rotation=True, flip=True, intensity_shift=False, noise=True):
    """
    Perform data augmentation on 3D MRI patches and masks, and increase the number of patches.

    Args:
    - mask_patches (numpy array): Array of shape (N_patches, 1, NX, NY, NZ).
    - input_patches (numpy array): Array of shape (N_patches, C, NX, NY, NZ), where C is 1 or 2 (input channels).
    - num_augmented_per_patch (int): Number of augmented patches to generate per original patch.
    - rotation (bool): Whether to apply random rotations.
    - flip (bool): Whether to apply random flips.
    - intensity_shift (bool): Whether to apply random intensity shifts (only to input channels).
    - noise (bool): Whether to add random Gaussian noise (only to input channels).

    Returns:
    - expanded_input_patches (numpy array): Array of augmented input patches.
    - expanded_mask_patches (numpy array): Array of augmented mask patches.
    """

    N_patches, C, NX, NY, NZ = input_patches.shape

    expanded_input_patches = []
    expanded_mask_patches = []

    for i in range(N_patches):
        original_input_patch = input_patches[i]
        original_mask_patch = mask_patches[i]

        for _ in range(num_aug):
            input_patch = np.copy(original_input_patch)
            mask_patch = np.copy(original_mask_patch)

            # Random rotation (90-degree increments to preserve shape)
            if rotation:
                axes = [(0, 1), (0, 2), (1, 2)]  # Axes are (X, Y), (X, Z), (Y, Z)
                for axis in axes:
                    if random.random() > 0.5:  # 50% chance of rotation
                        k = random.choice([1, 2, 3])  # Rotate 90, 180, or 270 degrees
                        mask_patch[0] = np.rot90(mask_patch[0], k=k, axes=axis)
                        for m in range(C):
                            input_patch[m] = np.rot90(input_patch[m], k=k, axes=axis)

            # Random flip
            if flip:
                for axis in range(0, 2):  # Axes 2, 3, 4 correspond to X, Y, Z
                    if random.random() > 0.5:  # 50% chance of flip
                        mask_patch[0] = np.flip(mask_patch[0], axis=axis)
                        for m in range(C):
                            input_patch[m] = np.flip(input_patch[m], axis=axis)

            # Random intensity shift (only for input channels)
            if intensity_shift:
                # TODO : not on the patch
                shift = random.uniform(-0.1, 0.1)  # Random intensity shift
                input_patch += shift
                input_patch = np.clip(input_patch, 0, 1)  # Assuming normalized patches

            # Add Gaussian noise (only for input channels)
            if noise:
                noise_val = np.random.normal(0, 0.01, input_patch.shape)  # Mean 0, Std 0.01
                input_patch += noise_val
                input_patch = np.clip(input_patch, 0, 1)  # Assuming normalized patches


            expanded_input_patches.append(input_patch)
            expanded_mask_patches.append(mask_patch)

    # Convert lists to numpy arrays
    expanded_input_patches = np.array(expanded_input_patches)
    expanded_mask_patches = np.array(expanded_mask_patches)

    masks = np.concatenate((mask_patches, expanded_mask_patches),0)
    images = np.concatenate((input_patches, expanded_input_patches),0)
    
    return masks, images