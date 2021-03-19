import cv2
import os

base_path = "dataset/jpeg/"
new_path = "dataset/"

#convert tiff files to jpg to run on model
for infile in os.listdir(base_path):
    print ("file : " + infile)
    read = cv2.imread(base_path + infile)
    outfile = infile.split('.')[0] + '.jpg'
    cv2.imwrite(new_path+outfile,read,[int(cv2.IMWRITE_JPEG_QUALITY), 200])