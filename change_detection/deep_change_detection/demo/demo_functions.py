#%pip install tifffile imagecodecs
# Python packages - if any are not installed use line above or "pip install <package_name>" in terminal

import fiona
import IPython
import ipywidgets
import ipyleaflet
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%matplotlib inline
import numpy as np
import random
import os
import tensorflow as tf
import geopandas as gpd
import importlib
import tifffile
import imagecodecs
from shutil import copyfile
from skimage import io
from tqdm import tqdm as tqdm
from pandas import read_csv
from math import floor, ceil, sqrt, exp
import time
from pprint import pprint
import geojson

from shapely.geometry import Polygon
import descarteslabs as dl
import descarteslabs.workflows as wf
from descarteslabs.vectors import FeatureCollection, Feature, properties as p

# import PyTorch and model functions
#PyTorch
from packaging import version
import torch
#if version.parse(torch.__version__) < version.parse("1.6.0"):
#    %pip install torch==1.6.0
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.transforms as tr

# Models
from change_detection.deep_change_detection.demo.models.unet import Unet
# from siamunet_conc import SiamUnet_conc
# from siamunet_diff import SiamUnet_diff
# from fresunet import FresUNet
# from smallunet import SmallUnet
# from smallunet_attempt import Unet

# import custom functions from utils file
from change_detection.deep_change_detection.demo.utils_cd import generate_tiff_from_polygons, save_test_results, test

base_dir = os.getcwd()+'/change_detection/deep_change_detection/'

def assign_all_variables():
    # Variables
    # Define data origin
    locName = 'killick' # File name for area
    lat, long = 18.5113, -72.2936 # Lat/Long for center of area of interest
    mzoom = 16 # Zoom for interactive map
    newLocation = True # Will create new geojson and tiff files

    # Data labelling
    gjData = False # Location of gdb data (False if no labels, path to file if labels)
    dmgAss = base_dir+ 'demo/geojsons/' +locName+"Damage.geojson" # Location of geojson (created if newLocation True)
    if newLocation == True:
        os.system('mkdir {} && touch {}'.format(base_dir+ 'demo/geojsons/', dmgAss))
    area, defArea = 0.001, 0.0025 # area is radius in lat/long around point label to be considered, defArea is for the case of no labels and defines area box size

    # Imagery variables
    imgColl, cloudFraction = "sentinel-2:L1C", 0.1 # Image collection and thrshold for cloud cover
    bandNum = 1 # 0-[r,g,b], 1-[r,g,b,nir], 2-[r,g,b,cloud-mask,red-edge-2,red-edge-3,red-edge-4,nir,swir1,swir2]
    imgNum = 2 # Number of image dates
    #img_st, img_end = ['2017-08-15','2019-08-01'], ['2017-12-15','2019-12-01'] # Before and after image dates
    img_st, img_end = ['2018-09-01', '2019-09-01'], ['2018-12-31', '2019-12-31'] # Before and after image dates
    tifResolution, tifTilesize, tifPad = 10, 512, 0 # True if first time on location, Size of images defined within AOI determined by geojson
    types = ["red green blue","red green blue nir","red green blue cloud-mask red-edge-2 red-edge-3 red-edge-4 nir swir1 swir2"]
    bands = types[bandNum]
    
    # Data storage
    dataPath = base_dir +"demo/new/" # Path to save location for generated images

    # Model specifications
    LOAD_TRAINED = True # Load models (instead of re-training)
    newTest = True # False - Add to test files, True - Only test on current area
    TYPE = bandNum # Model type ~ band number
    modelWeights = base_dir+'weights/unet_final_weights.pth.tar' # Weights file from best trained model (ask Seb for latest)
    PATH_TO_TRAIN = base_dir+ 'demo/new/' # Path to downloaded training data - will look to remove need for this
    FP_MODIFIER = 1 # Tuning parameter, use 1 if unsure
    PATCH_SIDE = 32
    BATCH_SIZE, NORMALISE_IMGS, TRAIN_STRIDE, DATA_AUG = 8, True, int(PATCH_SIDE/2) - 1, False
    
    return locName, lat, long, mzoom, newLocation, gjData, dmgAss, area, defArea, imgColl, cloudFraction, bandNum, imgNum, img_st, img_end, tifResolution, tifTilesize, tifPad, types, bands, dataPath, LOAD_TRAINED, newTest, TYPE, modelWeights, PATH_TO_TRAIN, FP_MODIFIER, PATCH_SIDE, BATCH_SIZE, NORMALISE_IMGS, TRAIN_STRIDE, DATA_AUG

locName, lat, long, mzoom, newLocation, gjData, dmgAss, area, defArea, imgColl, cloudFraction, bandNum, imgNum, img_st, img_end, tifResolution, tifTilesize, tifPad, types, bands, dataPath, LOAD_TRAINED, newTest, TYPE, modelWeights, PATH_TO_TRAIN, FP_MODIFIER, PATCH_SIDE, BATCH_SIZE, NORMALISE_IMGS, TRAIN_STRIDE, DATA_AUG = assign_all_variables()
            
def make_polygon():
    features = []
    poly = Polygon([[long-defArea, lat-defArea], [long+defArea, lat-defArea], [long+defArea, lat+defArea], [long-defArea, lat+defArea], [long-defArea, lat-defArea]])
    #to line up with other tiles in the demo
    poly = Polygon([[-72.28405213680702,18.514949071981608], 
                    [-72.28394484844642,18.50711528191177], 
                    [-72.30164742794472,18.50697986162461], 
                    [-72.30176130442698,18.51506553158718], 
                    [-72.28405213680702,18.514949071981608]])
    features.append(geojson.Feature(properties={},geometry=poly))
    fc = geojson.FeatureCollection(features)
    with open(dmgAss, 'w') as f:
        geojson.dump(fc, f)
    
def generate_tiffs_separate_bands():
    for i in range(imgNum):
        out = dataPath+locName+"/imgs_"+str(i+1)+"/"
        generate_tiff_from_polygons(dmgAss,
                        products=imgColl,
                        bands=bands,
                        resolution=tifResolution,
                        tilesize=tifTilesize,
                        pad=tifPad,
                        start_datetime=img_st[i],
                        end_datetime=img_end[i],
                        out_folder=out,
                        )
   
    file = open(dataPath+'test.txt','w') if newTest else open(dataPath+'test.txt','a') # Open test.txt file
    bandNumbers = [[4,3,2], [4,3,2,8], [4,3,2,1,5,6,7,8,9,10]] # Set appropriate band numbers for Sentinel bands

    for i in range(len([name for name in os.listdir(out) if 'image' in name])): # Loop over tile number from location
        imgFolder = dataPath+locName+'_'+str(i) # Create directory for each tile in location
        if not os.path.exists(imgFolder): os.makedirs(imgFolder)
        file.write(','+locName+'_'+str(i)) if i>0 else file.write(locName+'_'+str(i)) # Write tile number to locations to be tested

        if not os.path.exists(imgFolder+"/cm/"): os.makedirs(imgFolder+"/cm/")
        targDest = imgFolder+"/cm/"+locName+'_'+str(i)+"-cm.tif"
        copyfile(dataPath+locName+"/imgs_1/target_"+str(i)+".tiff", targDest) # Copy target file to tile directory
        imgconv = io.imread(targDest)
        io.imsave(imgFolder+"/cm/cm.png", imgconv)

        for j in range(imgNum): # Loop over time points for images
            img = io.imread(dataPath+locName+"/imgs_"+str(j+1)+"/"+'image_'+str(i)+'.tiff')
            dest = imgFolder+"/imgs_"+str(j+1)
            if not os.path.exists(dest): os.makedirs(dest)

            for k in range(img.shape[2]): # Loop over bands in each image
                io.imsave(dest+"/B"+"{0:0=2d}".format(bandNumbers[bandNum][k])+".tif",img[:,:,k])

    file.close()        
    


# Functions

### Put in another file!

def adjust_shape(I, s):
    """Adjust shape of grayscale image I to s."""
    
    # crop if necesary
    I = I[:s[0],:s[1]]
    si = I.shape
    
    # pad if necessary 
    p0 = max(0,s[0] - si[0])
    p1 = max(0,s[1] - si[1])
    
    return np.pad(I,((0,p0),(0,p1)),'edge')
    

def read_sentinel_img(path):
    """Read cropped Sentinel-2 image: RGB bands."""
    im_name = os.listdir(path)[0][:-7]
    r = io.imread(path + im_name + "B04.tif")
    g = io.imread(path + im_name + "B03.tif")
    b = io.imread(path + im_name + "B02.tif")
    
    I = np.stack((r,g,b),axis=2).astype('float')
    
    if NORMALISE_IMGS:
        I = (I - I.mean()) / I.std()

    return I

def read_sentinel_img_4(path):
    """Read cropped Sentinel-2 image: RGB and NIR bands."""
    im_name = os.listdir(path)[0][:-7]
    r = io.imread(path + im_name + "B04.tif")
    g = io.imread(path + im_name + "B03.tif")
    b = io.imread(path + im_name + "B02.tif")
    nir = io.imread(path + im_name + "B08.tif")
    
    I = np.stack((r,g,b,nir),axis=2).astype('float')
    
    I = I.reshape(I.shape[0],I.shape[1],I.shape[2])
    
    if NORMALISE_IMGS:
        I = (I - I.mean()) / I.std()

    return I

def read_sentinel_img_leq20(path):
    """Read cropped Sentinel-2 image: bands with resolution less than or equals to 20m."""
    im_name = os.listdir(path)[0][:-7]
    
    r = io.imread(path + im_name + "B04.tif")
    s = r.shape
    g = io.imread(path + im_name + "B03.tif")
    b = io.imread(path + im_name + "B02.tif")
    nir = io.imread(path + im_name + "B08.tif")
    
    ir1 = adjust_shape(zoom(io.imread(path + im_name + "B05.tif"),2),s)
    ir2 = adjust_shape(zoom(io.imread(path + im_name + "B06.tif"),2),s)
    ir3 = adjust_shape(zoom(io.imread(path + im_name + "B07.tif"),2),s)
    nir2 = adjust_shape(zoom(io.imread(path + im_name + "B8A.tif"),2),s)
    swir2 = adjust_shape(zoom(io.imread(path + im_name + "B11.tif"),2),s)
    swir3 = adjust_shape(zoom(io.imread(path + im_name + "B12.tif"),2),s)
    
    I = np.stack((r,g,b,nir,ir1,ir2,ir3,nir2,swir2,swir3),axis=2).astype('float')
    
    if NORMALISE_IMGS:
        I = (I - I.mean()) / I.std()

    return I

def read_sentinel_img_leq60(path):
    """Read cropped Sentinel-2 image: all bands."""
    im_name = os.listdir(path)[0][:-7]
    
    r = io.imread(path + im_name + "B04.tif")
    s = r.shape
    g = io.imread(path + im_name + "B03.tif")
    b = io.imread(path + im_name + "B02.tif")
    nir = io.imread(path + im_name + "B08.tif")
    
    ir1 = adjust_shape(zoom(io.imread(path + im_name + "B05.tif"),2),s)
    ir2 = adjust_shape(zoom(io.imread(path + im_name + "B06.tif"),2),s)
    ir3 = adjust_shape(zoom(io.imread(path + im_name + "B07.tif"),2),s)
    nir2 = adjust_shape(zoom(io.imread(path + im_name + "B8A.tif"),2),s)
    swir2 = adjust_shape(zoom(io.imread(path + im_name + "B11.tif"),2),s)
    swir3 = adjust_shape(zoom(io.imread(path + im_name + "B12.tif"),2),s)
    
    uv = adjust_shape(zoom(io.imread(path + im_name + "B01.tif"),6),s)
    wv = adjust_shape(zoom(io.imread(path + im_name + "B09.tif"),6),s)
    swirc = adjust_shape(zoom(io.imread(path + im_name + "B10.tif"),6),s)
    
    I = np.stack((r,g,b,nir,ir1,ir2,ir3,nir2,swir2,swir3,uv,wv,swirc),axis=2).astype('float')
    
    if NORMALISE_IMGS:
        I = (I - I.mean()) / I.std()

    return I

def read_sentinel_img_trio(path):
    """Read cropped Sentinel-2 image pair and change map."""
#     read images
    if TYPE == 0:
        I1 = read_sentinel_img(path + '/imgs_1/')
        I2 = read_sentinel_img(path + '/imgs_2/')
    elif TYPE == 1:
        I1 = read_sentinel_img_4(path + '/imgs_1/')
        I2 = read_sentinel_img_4(path + '/imgs_2/')
    elif TYPE == 2:
        I1 = read_sentinel_img_leq20(path + '/imgs_1/')
        I2 = read_sentinel_img_leq20(path + '/imgs_2/')
    elif TYPE == 3:
        I1 = read_sentinel_img_leq60(path + '/imgs_1/')
        I2 = read_sentinel_img_leq60(path + '/imgs_2/')
        
    cm = io.imread(path + '/cm/cm.png', as_gray=True) != 0
    
    # crop if necessary
    s1 = I1.shape
    s2 = I2.shape
    print(s1,s2)
    I2 = np.pad(I2,((0, s1[0] - s2[0]), (0, s1[1] - s2[1]), (0,0)),'edge')
    
    
    return I1, I2, cm



def reshape_for_torch(I):
    """Transpose image for PyTorch coordinates."""
#     out = np.swapaxes(I,1,2)
#     out = np.swapaxes(out,0,1)
#     out = out[np.newaxis,:]
    out = I.transpose((2, 0, 1))
    return torch.from_numpy(out)



class ChangeDetectionDataset(Dataset):
    """Change Detection dataset class, used for both training and test data."""

    def __init__(self, path, train = True, patch_side = 96, stride = None, use_all_bands = False, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        # basics
        self.transform = transform
        self.path = path
        self.patch_side = patch_side
        if not stride:
            self.stride = 1
        else:
            self.stride = stride
        
        if train:
            fname = 'train.txt'
        else:
            fname = 'test.txt'
        
#         print(path + fname)
        self.names = read_csv(path + fname).columns
        self.n_imgs = self.names.shape[0]
        
        n_pix = 0
        true_pix = 0
        
        
        # load images
        self.imgs_1 = {}
        self.imgs_2 = {}
        self.change_maps = {}
        self.n_patches_per_image = {}
        self.n_patches = 0
        self.patch_coords = []
        for im_name in tqdm(self.names):
            # load and store each image
            print(im_name)
            I1, I2, cm = read_sentinel_img_trio(self.path + im_name)
            self.imgs_1[im_name] = reshape_for_torch(I1)
            self.imgs_2[im_name] = reshape_for_torch(I2)
            self.change_maps[im_name] = cm
            
            s = cm.shape
            n_pix += np.prod(s)
            true_pix += cm.sum()
            
            # calculate the number of patches
            s = self.imgs_1[im_name].shape
            n1 = ceil((s[1] - self.patch_side + 1) / self.stride)
            n2 = ceil((s[2] - self.patch_side + 1) / self.stride)
            n_patches_i = n1 * n2
            self.n_patches_per_image[im_name] = n_patches_i
            self.n_patches += n_patches_i
            
            # generate path coordinates
            for i in range(n1):
                for j in range(n2):
                    # coordinates in (x1, x2, y1, y2)
                    current_patch_coords = (im_name, 
                                    [self.stride*i, self.stride*i + self.patch_side, self.stride*j, self.stride*j + self.patch_side],
                                    [self.stride*(i + 1), self.stride*(j + 1)])
                    self.patch_coords.append(current_patch_coords)
                    
        self.weights = [ FP_MODIFIER * 2 * true_pix / n_pix, 2 * (n_pix - true_pix) / n_pix]   

    def get_img(self, im_name):
        return self.imgs_1[im_name], self.imgs_2[im_name], self.change_maps[im_name]

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        current_patch_coords = self.patch_coords[idx]
        im_name = current_patch_coords[0]
        limits = current_patch_coords[1]
        centre = current_patch_coords[2]
        
        I1 = self.imgs_1[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
        I2 = self.imgs_2[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
        
        label = self.change_maps[im_name][limits[0]:limits[1], limits[2]:limits[3]]
        label = torch.from_numpy(1*np.array(label)).float()
        
        sample = {'I1': I1, 'I2': I2, 'label': label}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

class RandomFlip(object):
    """Flip randomly the images in a sample."""

#     def __init__(self):
#         return

    def __call__(self, sample):
        I1, I2, label = sample['I1'], sample['I2'], sample['label']
        
        if random.random() > 0.5:
            I1 =  I1.numpy()[:,:,::-1].copy()
            I1 = torch.from_numpy(I1)
            I2 =  I2.numpy()[:,:,::-1].copy()
            I2 = torch.from_numpy(I2)
            label =  label.numpy()[:,::-1].copy()
            label = torch.from_numpy(label)

        return {'I1': I1, 'I2': I2, 'label': label}

class RandomRot(object):
    """Rotate randomly the images in a sample."""

#     def __init__(self):
#         return

    def __call__(self, sample):
        I1, I2, label = sample['I1'], sample['I2'], sample['label']
        
        n = random.randint(0, 3)
        if n:
            I1 =  sample['I1'].numpy()
            I1 = np.rot90(I1, n, axes=(1, 2)).copy()
            I1 = torch.from_numpy(I1)
            I2 =  sample['I2'].numpy()
            I2 = np.rot90(I2, n, axes=(1, 2)).copy()
            I2 = torch.from_numpy(I2)
            label =  sample['label'].numpy()
            label = np.rot90(label, n, axes=(0, 1)).copy()
            label = torch.from_numpy(label)

        return {'I1': I1, 'I2': I2, 'label': label}

"""
Network initialisation function
"""
# Initialise network
def init_network(DATA_AUG, PATH_TO_TRAIN, TRAIN_STRIDE, TYPE):
    if DATA_AUG:
        data_transform = tr.Compose([RandomFlip(), RandomRot()])
    else:
        data_transform = None
    train_dataset = ChangeDetectionDataset(PATH_TO_TRAIN, train = True, stride = TRAIN_STRIDE, transform=data_transform)
    weights = torch.FloatTensor(train_dataset.weights)

    # 0-RGB | 1-RGBIr | 2-All bands s.t. resulution <= 20m | 3-All bands

    if TYPE == 0:
        net, net_name = Unet(2*3, 2), 'FC-EF'
    #     net, net_name = Unet(2*3, 2), 'FC-EF'
    #     net, net_name = SiamUnet_conc(3, 2), 'FC-Siam-conc'
    #     net, net_name = SiamUnet_diff(3, 2), 'FC-Siam-diff'#
    #     net, net_name = FresUNet(2*3, 2), 'FresUNet'
    elif TYPE == 1:
    #     net, net_name = SmallUnet(2*4, 2), 'FC-EF'
        net, net_name = Unet(2*4, 2), 'FC-EF'
    #     net, net_name = SiamUnet_conc(4, 2), 'FC-Siam-conc'
    #     net, net_name = SiamUnet_diff(4, 2), 'FC-Siam-diff'
    #     net, net_name = FresUNet(2*4, 2), 'FresUNet'
    elif TYPE == 2:
        net, net_name = SmallUnet(2*10, 2), 'FC-EF'
    #     net, net_name = Unet(2*10, 2), 'FC-EF'
    #     net, net_name = SiamUnet_conc(10, 2), 'FC-Siam-conc'
    #     net, net_name = SiamUnet_diff(10, 2), 'FC-Siam-diff'
    #     net, net_name = FresUNet(2*10, 2), 'FresUNet'
    elif TYPE == 3:
        net, net_name = SmallUnet(2*13, 2), 'FC-EF'
    #     net, net_name = Unet(2*13, 2), 'FC-EF'
    #     net, net_name = SiamUnet_conc(13, 2), 'FC-Siam-conc'
    #     net, net_name = SiamUnet_diff(13, 2), 'FC-Siam-diff'
    #     net, net_name = FresUNet(2*13, 2), 'FresUNet'

    criterion = nn.NLLLoss(weight=weights) # to be used with logsoftmax output
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    
    return train_dataset, weights, net, net_name, criterion, params
    







# Load in new dataset
def load_and_run_model():
    new_dataset = ChangeDetectionDataset(dataPath, train = False, stride = TRAIN_STRIDE)
    new_loader = DataLoader(new_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 1)
    #return new_dataset, new_loader
    
    train_dataset, weights, net, net_name, criterion, params = init_network(DATA_AUG, PATH_TO_TRAIN, TRAIN_STRIDE, TYPE)
    #print('Trained parameters ', params)

    
    net.load_state_dict(torch.load(modelWeights, map_location=torch.device('cpu')),strict=False)
    print('LOAD OK')
    #else:
    t_start = time.time()
    save_test_results(new_dataset, net, net_name)
    t_end = time.time()
    #print('Elapsed time: {}'.format(t_end - t_start))

    results = test(new_dataset, net, criterion)
    print('Check out imagery results')  

#run functions
def run_model():
    make_polygon()
    generate_tiffs_separate_bands()
    load_and_run_model()
    
    output_dir = str(os.getcwd()+'/change_detection/deep_change_detection/demo'+'/results/')

    import sys
    from PIL import Image

    images = [Image.open(output_dir+x) for x in os.listdir(output_dir) if 'png' in x]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    new_im.save(output_dir+ 'output.png')
    
    return str(os.getcwd()+'/change_detection/deep_change_detection/demo'+'/results/')


def display_results(output):
    img = mpimg.imread(output+str([file for file in os.listdir(output) if 'output' in file][0]))
    sentinel_2 = mpimg.imread(os.getcwd()+'/exposure_quantification/GHS_S2_Haiti.png')
    resultsFig, resultsAx = plt.subplots(1,2, figsize = (30,30))
    resultsAx[0].imshow(sentinel_2)
    resultsAx[0].title.set_text('Sentinel RGB')
    resultsAx[1].imshow(img)
    resultsAx[1].title.set_text('Change')
    


         