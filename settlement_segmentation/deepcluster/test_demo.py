#imports
import os
import descarteslabs as dl
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import Polygon, MultiPolygon, mapping
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
try:
    import faiss
except ImportError:
    # esnure you are using this version due to a object depreciation that means \
    # the clustering does not run on the newer version
    os.system("pip3 install faiss-gpu==1.6.1 && conda install -c pytorch torchvision cudatoolkit=10.1 pytorch -y")
import faiss
try:
    import rioxarray as rxr
except ModuleNotFoundError: 
    os.system("pip install rioxarray")
import rioxarray as rxr

import settlement_segmentation.deepcluster.models as models

def inference(DATA, ARCH, LR, K):
    
    base = '/settlement_segmentation/deepcluster/output/sentinel/'
    outfile= 'lr:{}_k:{}/'.format(LR, K)
    EXP= base+outfile

    model = models.__dict__[ARCH](sobel=True)
    fd = int(model.top_layer.weight.size()[1])
    model.top_layer = None
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    cudnn.benchmark = True

    # create optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=LR,
        momentum=0.9,
        weight_decay=10**-5,
    )

    # define loss function
    criterion = nn.CrossEntropyLoss().cuda()
    
    checkpoint = torch.load(EXP+'checkpoint.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    dev=torch.device("cuda") 
    model.to(dev)
    model.eval()

    # preprocessing of data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tra = [transforms.Resize(224),
           transforms.ToTensor(),
           normalize]
    
    dataset = datasets.ImageFolder(DATA, transform=transforms.Compose(tra))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    count = 0
    index = []
    columns = ['Informal', 'Cluster', 'geometry']
    df = pd.DataFrame(columns = columns) 
    
    with torch.no_grad():
         for data in data_loader:    
            data = data[0].numpy()
            output = model(torch.tensor(data).to(dev))
            index.append(output.data.cpu().numpy().argmax())
            
            #create dataframe from tif files for test data
            tif_folder = "/settlement_segmentation/data/cloud_free/test_tifs/tif/"
            file = [i for i in os.listdir(tif_folder) if 'tif' in i][count]
            tif = rasterio.open(tif_folder+file)
            if file.startswith('inf'):
                is_inf = 'inf'
            else:
                is_inf = 'not inf'
            
            coordinates = transform_bounds(tif.crs, 'EPSG:4326', tif.bounds.left,
                                                                 tif.bounds.bottom,
                                                                 tif.bounds.right, 
                                                                 tif.bounds.top)
            left, bottom, right, top = coordinates
            Geometry = Polygon([[left, top], [right, top], [right ,bottom], [left, bottom], [left, top]])
            df.loc[len(df)]= [is_inf, index[count], Geometry]
            count+=1
            
    #convert dataframe to geopandas dataframe for ease of plotting   
    df = gpd.GeoDataFrame(df, geometry='geometry', crs = 4326)
    
    df.plot(df['Cluster'])
    df.plot(df['Informal'])
    
    df.plot(df['Cluster'])
    plt.xlim([-77.05,-77.15 ])
    plt.ylim([18.35,18.43])
    df.plot(df['Informal'])
    plt.xlim([-77.05,-77.15 ])
    plt.ylim([18.35,18.43])

    df.plot(df['Cluster'])
    plt.xlim([-76.85,-76.75])
    plt.ylim([17.95,18.025])
    df.plot(df['Informal'])
    plt.xlim([-76.85,-76.75])
    plt.ylim([17.95,18.025])

    df.plot(df['Cluster'])
    plt.xlim([-72.29,-72.350])
    plt.ylim([18.54,18.59])
    df.plot(df['Informal'])
    plt.xlim([-72.29,-72.350])
    plt.ylim([18.54,18.59])
    
    return df, model
