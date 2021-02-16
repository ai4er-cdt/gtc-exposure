import tensorflow as tf
import descarteslabs as dl
import numpy as np
import json
import os
import datetime
import tifffile
import torch
import warnings

from shapely.geometry import shape
from osgeo import gdal, ogr
from typing import Sequence
from PIL import Image
from tqdm import tqdm as tqdm
from torch.autograd import Variable
from skimage import io
from math import floor, ceil, sqrt, exp

"""
Functions for generating tiff files
"""

numpy_dtype_to_gdal = {
    np.dtype("bool"): gdal.GDT_Byte,
    np.dtype("byte"): gdal.GDT_Byte,
    np.dtype("uint8"): gdal.GDT_Byte,
    np.dtype("uint16"): gdal.GDT_UInt16,
    np.dtype("int16"): gdal.GDT_Int16,
    np.dtype("uint32"): gdal.GDT_UInt32,
    np.dtype("int32"): gdal.GDT_Int32,
    np.dtype("float32"): gdal.GDT_Float32,
    np.dtype("float64"): gdal.GDT_Float64,
    "bool": gdal.GDT_Byte,
    "byte": gdal.GDT_Byte,
    "uint8": gdal.GDT_Byte,
    "uint16": gdal.GDT_UInt16,
    "int16": gdal.GDT_Int16,
    "uint32": gdal.GDT_UInt32,
    "int32": gdal.GDT_Int32,
    "float32": gdal.GDT_Float32,
    "float64": gdal.GDT_Float64,
    "uint": gdal.GDT_UInt16,
    "int": gdal.GDT_Int32,
    "float": gdal.GDT_Float64,
}

def gdal_dataset_from_geocontext(
    ctx: dict,
    n_bands: int,
    driver_name: str = "MEM",
    savename: str = "",
    dtype: str = "byte",
    options: Sequence = None,
):
    """Get a GDAL dataset using geocontext returned by dl.scenes.search.
    The output GDAL dataset will have the proper geo metdata, but
    won't contain raster data.  To do that, use gdal_dataset_from_narray.
    Parameters
    ----------
    ctx: dict
        Geocontext as returned by dl.scenes.search(...)
    n_bands: int
        The number of raster bands for the output dataset.
        You must specify manually, because the data product you're trying to
        save might have more or fewer bands than the original image.
    driver_name: str (optional)
        gdal driver name. Eg: MEM or GTiff
    savename: str (optional)
        Path to save dataset, if saving is desired.
    dtype: str (optional)
        Numpy style datatype for the dataset
    options: list (optional)
        A list of gdal dataset options like ['COMPRESS=LZW']

    Returns
    -------
    ds: gdal.Dataset
        The output dataset
    """
    options = options or []

    n_rows = ctx.tilesize
    n_cols = ctx.tilesize
    gdal_dtype = numpy_dtype_to_gdal[dtype]
    driver = gdal.GetDriverByName(driver_name)
    ds = driver.Create(savename, n_rows, n_cols, n_bands, gdal_dtype, options=options)
    # Grab projection and geotransform from metadata.
    proj_wkt = ctx.wkt
    ds.SetProjection(proj_wkt)
    ds.SetGeoTransform(ctx.geotrans)

    return ds

def generate_tiff_from_polygons(
    polygons,
    products,
    bands,
    resolution,
    tilesize,
    pad,
    start_datetime,
    end_datetime,
    out_folder,
    newLoc,
    seed=0,
    debugging = False
 ):
    if isinstance(polygons, str):
        fname = polygons
        with open(fname) as f:
            poly = json.load(f)
        polygons = {"type": "FeatureCollection", "features": poly["features"]}
        ogr_ds = ogr.Open(fname)
    elif isinstance(polygons, dict):
        assert polygons["type"] == "FeatureCollection"
        ogr_ds = ogr.Open(json.dumps(polygons))
    else:
        print("Wrong datatype for POLYGONS.")
        return 0
    n_features = len(polygons["features"])

    layer = ogr_ds.GetLayer()
    
    #Bounding box covering all the polygons
    poly_coords = []
    for i in range(n_features):
        poly_coords.extend(polygons['features'][i]['geometry']['coordinates'][0])
        
    min_corner = [min(map(lambda x:x[0], poly_coords)), min(map(lambda x:x[1], poly_coords))]
    max_corner = [max(map(lambda x:x[0], poly_coords)), max(map(lambda x:x[1], poly_coords))]
    
    #Turn this into a polygon
    aoi_feature = {
        "type":"Polygon",
        "coordinates":[[
            min_corner,
            [min_corner[0], max_corner[1]],
            max_corner,
            [max_corner[0], min_corner[1]],
            min_corner
        ]]
    }  

    #Get dltiles for bounding box around geojsons
    dltiles = dl.scenes.DLTile.from_shape(shape = aoi_feature, resolution = resolution, tilesize = tilesize, pad = pad)
    
    if not newLoc:
      return dltiles
    
    arrs = []
    trgts = []
    
    #Normalisation for spectral bands for the given product. ASSUMES THEY ARE ALL ON THE SAME SCALE
    product_bands = list(dl.catalog.Product.get(products).bands())
    
    max_spectral_val = -1
    for band in product_bands:
        if band.type == 'spectral':
            max_spectral_val = band.data_range[1]
            break
    
    if max_spectral_val == -1:
        print("Could not find maximum spectral value for bands in product")
        return
    
    if not os.path.exists(out_folder):
                os.makedirs(out_folder)
    for i, dltile in enumerate(dltiles):
        savefile_image = out_folder + "{}.tiff".format("image_"+str(i))
        savefile_target = out_folder + "{}.tiff".format("target_"+str(i))
        
        # Search for scenes for the dltile
        scenes, ctx = dl.scenes.search(
            aoi=dltile,
            products=products,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
        )
        
        #Break if no valid scene for this dltile
        if len(scenes) == 0:
            break
            
        #Get the raster image for these scenes - i.e the satellite data in the given bands
        arr = scenes.mosaic(bands=bands, ctx=ctx, bands_axis=-1) #* 255 / max_spectral_val
        #n.b you can plot this using plt.imshow(arr.data)
        arr = np.ma.MaskedArray.astype(np.rint(arr/max_spectral_val * 255), np.uint8)
        tifffile.imsave(savefile_image, arr, planarconfig="contig")
        #image = Image.fromarray(arr)
        #image.save(savefile_image)
        
        # Using the metadata get the target - the map of whether the settlements are informal or not
        ds_target = gdal_dataset_from_geocontext(
            ctx,
            1,
            driver_name="GTiff",
            savename=savefile_target,
            dtype="byte",
            options=["COMPRESS=LZW"],
        )
        
        gdal.RasterizeLayer(
            ds_target,
            [1],
            layer,
            burn_values=[1],
            options=["ALL_TOUCHED=TRUE", "COMPRESS=LZW"],
        )

        del ds_target
      
        if debugging:
            arrs.append(arr)
            img_target = np.array(Image.open(savefile_target))
            trgts.append(img_target)
            #Again can be viewed with plt.imshow(img_target)
            
    if debugging: 
        return arrs, trgts
    else:
        return dltiles

      
def save_test_results(dset, net, net_name):
    for name in tqdm(dset.names):
        with warnings.catch_warnings():
            I1, I2, cm = dset.get_img(name)
            I1 = Variable(torch.unsqueeze(I1, 0).float())#.cuda()
            I2 = Variable(torch.unsqueeze(I2, 0).float())#.cuda()
            out = net(I1, I2)
            _, predicted = torch.max(out.data, 1)
            I = np.stack((255*cm,255*np.squeeze(predicted.cpu().numpy()),255*cm),2)
            if not os.path.exists('./results'):
                os.makedirs('./results')
            io.imsave(f'./results/{net_name}-{name}.png',I)
            
L = 1024

def kappa(tp, tn, fp, fn):
    N = tp + tn + fp + fn
    p0 = (tp + tn) / N
    pe = ((tp+fp)*(tp+fn) + (tn+fp)*(tn+fn)) / (N * N)
    
    return (p0 - pe) / (1 - pe)

def test(dset, net, criterion):
    net.eval()
    tot_loss = 0
    tot_count = 0
    tot_accurate = 0
    
    n = 2
    class_correct = list(0. for i in range(n))
    class_total = list(0. for i in range(n))
    class_accuracy = list(0. for i in range(n))
    
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    for img_index in tqdm(dset.names):
        I1_full, I2_full, cm_full = dset.get_img(img_index)
        
        s = cm_full.shape
        
        for ii in range(ceil(s[0]/L)):
            for jj in range(ceil(s[1]/L)):
                xmin = L*ii
                xmax = min(L*(ii+1),s[1])
                ymin = L*jj
                ymax = min(L*(jj+1),s[1])
                I1 = I1_full[:, xmin:xmax, ymin:ymax]
                I2 = I2_full[:, xmin:xmax, ymin:ymax]
                cm = cm_full[xmin:xmax, ymin:ymax]

                I1 = Variable(torch.unsqueeze(I1, 0).float())#.cuda()
                I2 = Variable(torch.unsqueeze(I2, 0).float())#.cuda()
                cm = Variable(torch.unsqueeze(torch.from_numpy(1.0*cm),0).float())#.cuda()

                output = net(I1, I2)
                    
                loss = criterion(output, cm.long())
                tot_loss += loss.data * np.prod(cm.size())
                tot_count += np.prod(cm.size())

                _, predicted = torch.max(output.data, 1)

                c = (predicted.int() == cm.data.int())
                for i in range(c.size(1)):
                    for j in range(c.size(2)):
                        l = int(cm.data[0, i, j])
                        class_correct[l] += c[0, i, j]
                        class_total[l] += 1
                        
                pr = (predicted.int() > 0).cpu().numpy()
                gt = (cm.data.int() > 0).cpu().numpy()
                
                tp += np.logical_and(pr, gt).sum()
                tn += np.logical_and(np.logical_not(pr), np.logical_not(gt)).sum()
                fp += np.logical_and(pr, np.logical_not(gt)).sum()
                fn += np.logical_and(np.logical_not(pr), gt).sum()
        
    net_loss = tot_loss/tot_count        
    net_loss = float(net_loss.cpu().numpy())
    
    net_accuracy = 100 * (tp + tn)/tot_count
    
    for i in range(n):
        class_accuracy[i] = 100 * class_correct[i] / max(class_total[i],0.00001)
        class_accuracy[i] =  float(class_accuracy[i].cpu().numpy())

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    dice = 2 * prec * rec / (prec + rec)
    prec_nc = tn / (tn + fn)
    rec_nc = tn / (tn + fp)
    
    pr_rec = [prec, rec, dice, prec_nc, rec_nc]
    
    k = kappa(tp, tn, fp, fn)
    
    return {'net_loss': net_loss, 
            'net_accuracy': net_accuracy, 
            'class_accuracy': class_accuracy, 
            'precision': prec, 
            'recall': rec, 
            'dice': dice, 
            'kappa': k}