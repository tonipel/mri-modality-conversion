# mri-modality-conversion

## Docker setup

### Install Docker

Download and install Docker (https://www.docker.com/).

### Run source code

Run following commands in the source code directory.

Build the docker image:
```
docker build -t mri-modality-conversion .
```

Execute the main file within the docker image:
```
docker run mri-modality-conversion python3 /code/pix2pix.py <arguments>
```

Arguments could be for example:
```
docker run mri-modality-conversion python3 /code/pix2pix.py -f data -m convert -c t1_to_t2
```







## Under Development

To make the conversion pipeline more streamlined, the development for a conversion module for .png to Nifti (.nii) format has been started in ```mri-modality-conversion/png2nii/```. While not fully developed for production, the module is ready for testing different .png directories and .nii test files.

### Modules

#### ```png2nii.py```

Functions for running and testing the .png to .nii conversion. Contains two functions:

* ```convert_to_nifti()``` : Takes a stack of .png images in a directory and saves them into a .nii file format


* ```show_sample_slices()``` : Used for visually observing the contents of the .nii file. From an input .nii file, displays three randomly chosen slices (one from each 3D axis).


### Used external packages

For production use, these can (and probably should) be incorporated into the environment build.

```- numpy=1.19.2 
- nibabel=3.2.1
- pillow=8.0.1
- matplotlib=3.3.2

