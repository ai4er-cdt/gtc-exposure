import os
import sys
import numpy as np
import skimage.exposure as exposure
import torchvision.datasets as datasets
import torchvision.transforms as transforms
try:
    import tiffile
except ModuleNotFoundError:
    os.system('pip install tiffile')
import tiffile
try:
    import cv2
except ModuleNotFoundError:
    os.system('pip install opencv-python')
import cv2


base_path = sys.argv[1]
new_path = '/'.join(base_path.split('/')[:-2])+"/jpeg/"

os.system('mkdir {}'.format(new_path))

#convert tiff files to jpg to run on model
for infile in os.listdir(base_path):
    if infile.endswith(".tif"):
        print ("file : " + infile)
        s2_cloud_free = tiffile.imread(base_path + infile)
        #remove images on the edges where some of the tile is cut off
        if s2_cloud_free.min()>= 0.00001: 
            outfile = infile.split('.')[0] + '.jpg'
            #normalise images
            s2_cloud_free_norm = exposure.rescale_intensity(s2_cloud_free,
                                                            in_range='image',
                                                            out_range=(0,255)).astype(np.uint8)
            cv2.imwrite(new_path+outfile,s2_cloud_free_norm,[int(cv2.IMWRITE_JPEG_QUALITY), 200])
