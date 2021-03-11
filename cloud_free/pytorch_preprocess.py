import os
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

base_path = "/home/jovyan/gtc-exposure/cloud_free/cut_images/"
new_path = "/home/jovyan/gtc-exposure/cloud_free/train_images/jpeg/"

os.system('mkdir {}'.format(new_path))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

#convert tiff files to jpg to run on model
for infile in os.listdir(base_path):
    print ("file : " + infile)
    read = tiffile.imread(base_path + infile)
    read = read[:224,:224]
    outfile = infile.split('.')[0] + '.jpg'
    cv2.imwrite(new_path+outfile,read,[int(cv2.IMWRITE_JPEG_QUALITY), 200])
