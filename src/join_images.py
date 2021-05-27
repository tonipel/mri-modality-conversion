import os
from PIL import Image

PATH = './imagedata/'
COMBINED_DIR = './combined/'

def main():
    '''
    This is an utility function which creates image pairs,
    that are concatenated horizontally.
    
    Global variables:
        PATH = data directory containing input images
        COMBINED_DIR = output directory, where joined images are saved
    
    PATH directory must have subfolders named A and B,
    where A contains T1 images and B contains T2 images.
    '''
    if not os.path.isdir(COMBINED_DIR):
        os.mkdir(COMBINED_DIR) 

    dir_A = os.listdir(PATH + 'A')
    dir_B = os.listdir(PATH + 'B')

    for img_A, img_B in zip(dir_A, dir_B):
        images = [Image.open(x) for x in [PATH + 'A/' + img_A, PATH + 'B/' + img_B]]
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0]

        new_im.save(COMBINED_DIR + img_A)

if __name__ == "__main__":
    main()
