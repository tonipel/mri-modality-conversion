from png2nii import convert_to_nifti, show_sample_slices
import os
import nibabel as nib


def main():
    current_dir = os.getcwd()
    png_dir = os.path.join(current_dir, 'png2nii', 'png-imgs')         #Specify png images folder here
    dst_filename = 'png2nii_test.nii'

    "Reading a sample header"
    sample_nii = os.path.join(current_dir, 'png2nii' ,'rt1.nii')
    img_header = nib.load(sample_nii).header

    convert_to_nifti(png_dir, dst_filename, img_header)
    
    " (show sample slices) "
    #sample_nii2 = os.path.join(current_dir, 'png2nii' ,'png2nii_test.nii')
    #show_sample_slices(sample_nii2)
    
if __name__ == '__main__':
    main()