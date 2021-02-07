import os
import sys
from PIL import Image

PATH = './imagedata/'
COMBINED_DIR = './combined/'

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

