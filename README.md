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


## Usage of Pix2Pix image conversion tool

Pix2pix can be used for both training and converting the images based on the saved model.

### Different options to run python code

The different options to run code can be found by using command ```-h``` or ```--help```:
````
pix2pix.py -h 
````
This outputs following list of options:
````
List of options

  -h, --help            
                        show this help message and exit

  -f INPUT, --file INPUT
                        Path to input folder

  -m {train,convert}, --mode {train,convert}
                        Mode to run MRI image converison

  -c {t1_to_t2,t2_to_t1}, --convert {t1_to_t2,t2_to_t1}
                        Image translation direction

  -o OUTPUT, --output OUTPUT
                        Path to output folder

  -e EPOCHS, --epochs EPOCHS
                        Number of epochs for training

  -s SAVE, --save SAVE  
                        Path to save model

  -l LOAD, --load LOAD  
                        Load saved model from custom directory

  -cm {rgb,bw}, --colormode {rgb,bw}
                        Color mode of images, either RGB (rgb) or black white (bw)
````

### How to train network:

Train the model to convert T1 to T2 image
````
pix2pix.py -f {path_training_data} -m train -c t1_to_t2 -e {number_of_epochs} -s {path_save_model}
````

Train the model to convert T2 to T1 image
````
pix2pix.py -f {path_training_data} -m train -c t2_to_t1 -e {number_of_epochs} -s {path_save_model}
````

### How to convert images

Convert colored T1 image to colored T2 image by using saved model
````
pix2pix.py -f {path_convert_data} -m convert -c t1_to_t2 -o {path_converted_T2_images} -l {path_saved_model} -cm rgb
````

Convert black and white T2 image to balck and white T1 image by using saved model
````
pix2pix.py -f {path_convert_data} -m convert -c t2_to_t1 -o {path_converted_T1_images} -l {path_saved_model} -cm bw
````




## Under Development

To make the conversion pipeline more streamlined, the development for a conversion module for .png to Nifti (.nii) format has been started in ```mri-modality-conversion/png2nii/```. While not fully developed for production, the module is ready for testing different .png directories and .nii test files.

### Modules

#### ```png2nii.py```

Functions for running and testing the .png to .nii conversion. Contains two functions:

* ```convert_to_nifti()``` : Takes a stack of .png images in a directory and saves them into a .nii file format


* ```show_sample_slices()``` : Used for visually observing the contents of the .nii file. From an input .nii file, displays three randomly chosen slices (one from each 3D axis).


#### ```test_conversion.py```

A ```main()```- module for performing test conversions of .png stacks to nifti. It imports the functions from ```png2nii.py``` and implements the conversion after reading the directory path for the .png slices. There is also a commented out block of code (lines 17-19), 

```
    " (show sample slices) "
    #sample_nii2 = os.path.join(current_dir, 'png2nii' ,'png2nii_test.nii')
    #show_sample_slices(sample_nii2)
```

, which displays random slices from the newly formed .nii file. This can be done by removing the comments on these lines.



#### ```png2nii.ipynb```

A Jupyter Notebook for playing around/investigating different attributes of Nifti files and conversion techniques. As mentioned before, this feature is still under development, and specific attributes of code chunks are easier to test with .ipynb code cells. The purpose of the notebook is  **not** to implement any final features, but to test out or print code chunks that could be transferred to the conversion module itself.

### Used external packages

For production use, these can (and probably should) be incorporated into the environment build.

```
- numpy=1.19.2 
- nibabel=3.2.1
- pillow=8.0.1
- matplotlib=3.3.2
```


### Problems and points of future improvement

### 1. Pipeline implementation

At the moment, the aspects of our AI pipeline are somewhat disjoint. Ideally, one would want to perform all of the needed processes with one pipeline, one command, and one (command line) interface. Hence, a functionality that is for future development is the *automated* .png to .nii conversion to the model output images. This would help streamline the process.

### 2. Inaccuracies and artefacts in converted Nifti outputs

The tested converted .nii file has some gray area and other inaccurate artefacts some slices were visualized. This is something that could possibly be improved by adjusting the metadata that is inputted for the ```Nifti1Image```-generator during image conversion. For more details, see the test output images in ```png2nii.ipynb```. 


## License
[APACHE 2.0](https://www.apache.org/licenses/LICENSE-2.0/)
