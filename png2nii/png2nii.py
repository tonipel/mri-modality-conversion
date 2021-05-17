import os
import numpy as np
import nibabel as nib   #Check that you have nibabel
import sys
from PIL import Image
import matplotlib.pyplot as plt


def convert_to_nifti(png_dir, dst_filename, header):
    
    """
    Converts a directory of .png images into a .nii file
        Parameters:
            png_dir (str): Path to the directory containing the images used in convesion
            dst_filename (str): Name of the saved .nii file
            header (Nifti1Header object): Contains the header information used for the .nii file.
    
        Returns:
            Saved file in current working directory
    """
    
    "First need to open png folder"
    
    png_imgs = sorted(os.listdir(png_dir))
    num_slices = len(png_imgs) 

    "Read Images into an array"
    
    open_img = Image.open
    samp_img = open_img(f'{png_dir}/{png_imgs[0]}')
    full_arr = np.expand_dims(samp_img,2)

    open_img = Image.open
    for img_name in png_imgs[1:]:
        img_png = open_img(f'{png_dir}/{img_name}')
        img_arr = np.expand_dims(np.asarray(img_png), 2)
        full_arr = np.concatenate((full_arr, img_arr), axis = 2)

    "Save to nifti"
    
    affine = header.get_best_affine()
    img_nii = nib.Nifti1Image(full_arr, header=header, affine=affine)
    nib.save(img_nii, dst_filename)



'''Picks random slices and displays them.
- Sample code assumed from https://nipy.org/nibabel/coordinate_systems.html'''

def show_sample_slices(nii_file):
    
    """ Function to display row of image slices 
            
        Parameters:
            nii_file (.nii): Nifti file to choose slices from
            
        Returns:
            Display of three sampled slices
    """
    
    img = nib.load(nii_file)
    img_data =  img._dataobj
    data_shape = img_data.shape
    
    id1, id2, id3 = np.random.uniform(size=3, low=0, high=data_shape[0])
    
    slices = [img_data[id1,:,:], img_data[:,id2,:], img_data[:,:,id3]]
    
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
    plt.show()