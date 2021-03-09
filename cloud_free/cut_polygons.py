import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import gdal
import pyproj
import fiona
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_bounds
from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.ops import transform

try:
    import rioxarray as rxr
except ModuleNotFoundError: 
    os.system("pip install rioxarray")
import rioxarray as rxr


def retrieve_image(image_dir, polygon):
    
    columns = ['Filename', 'Bands', 'Width', 'Height', 'Coordinates']
    dataframe = pd.DataFrame(columns = columns)
    
    for file in os.listdir(image_dir):
        if file.endswith('.tiff') or file.endswith('.tif'):
                dataset = rasterio.open(image_dir+ file)
    
                left= dataset.bounds.left
                bottom = dataset.bounds.bottom
                right = dataset.bounds.right
                top = dataset.bounds.top

                coordinates = transform_bounds(dataset.crs, 'EPSG:4326', left, bottom, right, top)
                left, bottom, right, top = coordinates
                Geometry = (Polygon([ [left, top],
                                      [right, top],
                                      [right ,bottom],
                                      [left, bottom]]))
    
                dataframe.loc[len(dataframe)]= [file, dataset.count, dataset.width, dataset.height, Geometry]
    
    interceptions = []
    for i in range(len(dataframe)):
        if dataframe.loc[i]['Coordinates'].intersects(polygon):
            row = [i for i in dataframe.loc[i]]
            file = row[0]
            interceptions.append(file)
            print('Filename: {} \nBands: {} \nWidth: {} \nHeight: {} \nCoordinates: {}'.format(file,
                                                                                               row[1],
                                                                                               row [2],
                                                                                               row[3],
                                                                                               row[4]))
            
            
    project = pyproj.Transformer.from_proj(
        pyproj.Proj(init='epsg:4326'), # source coordinate system
        pyproj.Proj(init='epsg:32617'))
    poly = transform(project.transform, polygon)
            

    schema = {
        'geometry': 'Polygon',
        'properties': {'id': 'int'},
    }

    os.system("mkdir {}".format('poly_shapes'))
    
    with fiona.open('poly_shapes/'+'poly.shp', 'w', 'ESRI Shapefile', schema) as c:
        ## If there are multiple geometries, put the "for" loop here
        c.write({
            'geometry': mapping(poly),
            'properties': {'id': 123},
        })
        
    aoi = os.path.join('poly_shapes', "poly.shp")
    poly = gpd.read_file(aoi)
    
    s2_cloud_free = []
    for file in interceptions:
        img = rxr.open_rasterio(image_dir+file, masked=True).squeeze()

        s2_cloud_free.append(img.rio.clip(poly.geometry.apply(mapping)))
    
    return s2_cloud_free

def cut_polygon(in_path, out_path, polygons):

    os.system("mkdir {}".format(out_path))
    
    count = 0
    for polygon in polygons:
        s2_cloud_free_array = retrieve_image(in_path, polygon)

        for s2_cloud_free in s2_cloud_free_array:
            with rasterio.open(
                '{}train_image_{}.tif'.format(out_path, count),
                'w',
                driver='GTiff',
                height=s2_cloud_free.shape[1],
                width=s2_cloud_free.shape[2],
                count=s2_cloud_free.shape[0],
                dtype=s2_cloud_free.dtype,
                crs='epsg:32617'
                ) as dst:
                dst.write(s2_cloud_free)

            red = s2_cloud_free[0]/s2_cloud_free.max()
            green = s2_cloud_free[1]/s2_cloud_free.max()
            blue = s2_cloud_free[2]/s2_cloud_free.max()
            s2_cloud_free_norm = np.dstack((red, green, blue))

            plt.figure(figsize=(20,10))
            plt.imshow(s2_cloud_free_norm)
            plt.show()
            count+=1
        
    os.system("rm -r {}".format('poly_shapes'))
    

def split_tiles(path, tile_size):
    
    count=0
    for file in os.listdir(path):
        if file.endswith('.tif'):
        
            output_filename = 'tile_{}_'.format(count)

            tile_size_x, tile_size_y = tile_size, tile_size

            ds = gdal.Open(path + file)
            band = ds.GetRasterBand(1)
            xsize = band.XSize
            ysize = band.YSize

            for i in range(0, xsize, tile_size_x):
                for j in range(0, ysize, tile_size_y):
                    com_string = "gdal_translate -of GTIFF -srcwin " + str(i)+ ", " + str(j) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + str(path) + str(file) + " " + str(path) + str(output_filename) + str(i) + "_" + str(j) + ".tif"
                    os.system(com_string)

            os.system("rm {}{}".format(path, file))
            count+=1  
