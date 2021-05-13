# This .py file supplies all the functions necessary for the image ratio demonstration .ipynb notebook.
# The functionalities in this .py file are explicitly written out in the individual thresholding and
# unet_classifier notebooks, which are considerably less nauseating than trying to sift through this file.

#--------------------------------------------------#
# Python libraries
import numpy as np
import ipywidgets
import random
import os
import IPython
import ipywidgets
import ipyleaflet
import json
import geopandas as gpd
import pandas as pd
import geojson
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Python library functions
from shapely.geometry import Polygon, Point
from tqdm import tqdm
from ipyleaflet import Map, GeoJSON, GeoData, LegendControl
from ipywidgets import FloatRangeSlider
from tensorflow.keras.models import load_model

# Descartes Labs
import descarteslabs as dl
import descarteslabs.workflows as wf
from descarteslabs.vectors import FeatureCollection, properties as p
from descarteslabs.workflows import GeoContext, Feature, FeatureCollection

# Custom functions
#from unet import UNet

#-------------------------------------------------------#

## Initialisation - Assigning locations and variables

base = os.getcwd()+ '/change_detection/ratio_method/'

# Function for location dropdown list
def chooseLocation(): 
  return ipywidgets.Dropdown(placeholder='Choose Location', 
                                 options=[('train - Roseau, Dominica',0), ('train - Abricots, Haiti',1),
                                          ('test - Jeremie, Haiti',2), ('test - Port Salut, Haiti',3),
                                          ('settlement - Bidonville KSV, Haiti',4), ('settlement - Parry Town, Jamaica',5),
                                          ('high resolution - Hidalgo, Texas',6),('New Location',7)],
                                 description='Location:',
                                 disabled=False)

# Having chosen a location from the dropdown list, this function will assign its specified variables
def assignVariables(locationFunc):
  location = locationFunc.value # Get location choice

  # If 'New location' has been chosen then we supply a text box into which the user can enter their variable values
  if location == 7:
    # Create widgets
    newLat = ipywidgets.FloatText(value=18.6486, description='Latitude (decimal):', disabled=False)
    newLon = ipywidgets.FloatText(value=-74.3058, description='Longitude (decimal):', disabled=False)
    newZoom = ipywidgets.IntText(value=16, description='Map zoom:', disabled=False)
    newStart1 = ipywidgets.Text(value='2016-07-01', placeholder='yyyy-mm-dd', description='Image 1 start date:', disabled=False)
    newEnd1 = ipywidgets.Text(value='2016-09-30', placeholder='yyyy-mm-dd', description='Image 1 end date:', disabled=False)
    newStart2 = ipywidgets.Text(value='2016-10-06', placeholder='yyyy-mm-dd', description='Image 2 start date:', disabled=False)
    newEnd2 = ipywidgets.Text(value='2017-07-06', placeholder='yyyy-mm-dd', description='Image 2 end date:', disabled=False)
    newSat = ipywidgets.Text(value="sentinel-2:L1C", placeholder='sentinel-2:L1C', description='Satellite:', disabled=False)
    newCloud = ipywidgets.FloatText(value=0.2, description='Cloud Fraction:', disabled=False)
    newThreshold = ipywidgets.FloatText(value=0.01, description='Default threshold:', disabled=False)
    newCap = ipywidgets.FloatText(value=0.1, description='Default Cap', disabled=False)
    newBand = ipywidgets.Dropdown(placeholder='Choose Location', 
                                 options=[('red',['red']), ('green',['green']), ('blue',['blue']),
                                          ('red,green,blue',['red','green','blue']), ('near IR',['nir'])],
                                 description='Imagery bands:', disabled=False)
    newRes = ipywidgets.IntText(value=10, description='Satellite resolution:', disabled=False)
    newMask = ipywidgets.Text(value="", placeholder='mask.geojson', description='Path to mask if using one:', disabled=False)
    
    # Display widgets
    display(newLat,newLon,newZoom,newStart1,newEnd1,newStart2,newEnd2,newSat,newCloud,newThreshold,newCap,newBand,newRes,newMask)
  
    # Assign widgets to variables dictionary
    variables = {'latitude':newLat,
                 'longitude':newLon,
                 'zoom':newZoom,
                 'st_date':[newStart1,newStart2],
                 'end_date':[newEnd1,newEnd2],
                 'satellite':newSat, # Descartes product name
                 'visual':['red','green','blue'], # Imagery bands for display purposes
                 'cloudFraction':newCloud,
                 'threshold':newThreshold,
                 'cap':newCap,
                 'bands':[newBand],
                 'kSize':1, # Pixel dilatation kernel size
                 'resolution':newRes,
                 'mask':newMask
                }
    
  # For example locations - assign pre-determined values
  else:
    # Values for default locations
    allLats = [15.3031,18.6486,18.6421,18.0967,18.5113,18.3923,26.2293] # Latitude
    allLons = [-61.3834,-74.3058,-74.1167,-73.9284,-72.2974,-77.1186,-98.0828] # Longitude
    allZooms = [15,16,15,15,16,16,16] # Map Zoom
    start1 = ['2017-08-15','2016-07-01','2016-07-01','2016-05-01','2018-09-01','2018-09-01','2016-07-01']
    start2 = ['2017-10-01','2016-10-06','2016-10-06','2016-10-06','2019-09-01','2019-09-01','2019-01-01']
    end1 = ['2017-09-15','2016-09-30','2016-09-30','2016-09-30','2018-12-31','2018-12-30','2016-12-31']
    end2 = ['2017-12-01','2017-07-06','2016-11-06','2016-12-06','2019-12-31','2019-12-30','2019-06-30']
    allCloudFraction = [0.05,0.2,0.8,0.1,0.8,0.05,0.05] # Cloud fraction for Sentinel image retrieval
    allThreshold = [0.001, 0.001,0.001,0.01,0.1,0.1,-0.5] # Initial bounds for logarithmic ratio change detection
    allCaps = [0.1,0.1,0.1,0.1,0.3,0.3,-0.1] # Initial bounds for logarithmic ratio change detection
    allBands = [['red'],['green'],['red'],['red','green','blue'],['red'],['red'],['red','green','blue']] # Bands used to evaluate ratio
    resolution = [10,10,10,10,10,10,1]
    satellite =  "usda:naip:rgbn:v1" if location is 6 else "sentinel-2:L1C"

    # Assign to variables dictionary
    variables = {'latitude':allLats[location],
                 'longitude':allLons[location],
                 'zoom':allZooms[location],
                 'st_date':[start1[location], start2[location]],
                 'end_date':[end1[location], end2[location]],
                 'satellite':satellite, # Descartes product name
                 'visual':['red','green','blue'], # Imagery bands for display purposes
                 'cloudFraction':allCloudFraction[location],
                 'threshold':allThreshold[location],
                 'cap':allCaps[location],
                 'bands':allBands[location],
                 'kSize':1, # Pixel dilatation kernel size
                 'resolution':resolution[location],
                 'mask':''
                } 
  
  # For locations with Copernicus EMS damage assessments
    if location < 4:
      # Values for default locations
      # Damage assessment database location (.dbf file needs .prj,.shp,.shx in same directory)
      allDmgAssess = [base + "gradings/EMSR246_04ROSEAU_02GRADING_v1_5500_settlements_point_grading.dbf",
                      base + "gradings/EMSR185_35ABRICOTS_02GRADING_v1_2500_settlements_point_grading.dbf", 
                      base + "gradings/EMSR185_11JEREMIE_02GRADING_MONIT01_v1_4000_settlements_point_grading.dbf",
                      base + "gradings/EMSR185_09PORTSALUT_02GRADING_v1_5500_settlements_point_grading.dbf"] 
      # Damage geojson with building footprints
      allDmgFiles = [base + 'RoseauDamage0004g3.geojson','HaitiAbricotsDamage0004g3.geojson',
                     base + 'HaitiJeremieDamage0004g3.geojson','HaitiPortSalutDamage0004g3.geojson'] 
      # Geojson file masking feature such as ocean
      allMaskPoly = ['coastlines/swDominicaOcean.geojson', base + 'coastlines/swHaitiCoastline.geojson', 
                     base + 'coastlines/swHaitiCoastline.geojson','coastlines/PortSalutCoastline.geojson'] 

      # Assign to dictionary
      variables['damageAssessment'], variables['damageGeojson'], variables['mask'] = allDmgAssess[location], 'geojsons/'+allDmgFiles[location], allMaskPoly[location]
      
      # Default variables same for all
      variables['grades'] = ['Completely Destroyed','Highly Damaged','Moderately Damaged'] # Options: 'Not Applicable','Negligible to slight damage', 'Moderately Damaged', 'Highly Damaged', 'Completely Destroyed'
      variables['area'] = 0.0004 # Building polygon size in lat/long degrees
    
    # Show variables button
    button = ipywidgets.Button(description="Show variables")
    output = ipywidgets.Output()
    display(button, output)
    def on_button_clicked(b):
        with output: print(variables)
    button.on_click(on_button_clicked)
    
  return variables, {}

# Once new variables are entered for new location - these are assigned to dictionary
def submitNewLocation(variables, updates):
  # If new location
  if not (type(variables['latitude']) == float) or not (updates == {}):
    updates = variables if not (type(variables['latitude']) == float) else updates
    variables = {'latitude':updates['latitude'].value,
                 'longitude':updates['longitude'].value,
                 'zoom':updates['zoom'].value,
                 'st_date':[updates['st_date'][0].value, updates['st_date'][1].value],
                 'end_date':[updates['end_date'][0].value, updates['end_date'][1].value],
                 'satellite':updates['satellite'].value, # Descartes product name
                 'visual':['red','green','blue'], # Imagery bands for display purposes
                 'cloudFraction':updates['cloudFraction'].value,
                 'threshold':updates['threshold'].value,
                 'cap':updates['cap'].value,
                 'bands':updates['bands'][0].value,
                 'kSize':1, # Pixel dilatation kernel size
                 'resolution':updates['resolution'].value,
                 'mask':updates['mask'].value
                }
    
  return variables, updates

#---------------------------------------------------#

## 1 - Visualise Imagery


def download_data():
    os.system('chmod +x change_detection/ratio_method/download_data.sh')
    os.system('./change_detection/ratio_method/download_data.sh')

# Function using GetImage to retrieve imagery for chosen before-after dates
def beforeAfterImages(variables):
  download_data()
  m1 = wf.interactive.MapApp()
  m1.center, m1.zoom = (variables['latitude'], variables['longitude']), variables['zoom']
  
  # Loop over dates (time 1 & ime 2)
  for i in range(len(variables['st_date'])):
      for j in variables['bands']: getImage(variables,i,j,m1,0,False) # Retrieve layer for each band to be used for ratio
      getImage(variables,i,variables['visual'],m1,1) # Retrieve layer for visual
  
  return m1

# Function which retrieves imagery for specified time period and band. Allow specification of layer opacity and map number
def getImage(v,time,bands,mapNum,opacity=1,visualise=True):
    img = wf.ImageCollection.from_id(v['satellite'],start_datetime=v['st_date'][time], end_datetime=v['end_date'][time])
    if 'sentinel' in v['satellite']: # Use sentinel cloud-mask band if available
        img = img.filter(lambda img: img.properties["cloud_fraction"] <= v['cloudFraction'])
        img = img.map(lambda img: img.mask(img.pick_bands('cloud-mask')==1))
    mos = (img.mosaic().pick_bands(bands))
    globals()['mos_'+str(time+1)+str(bands)] = mos
    if visualise:
      display = mos.visualize('Before', map=mapNum) if time < 1 else mos.visualize('After', map=mapNum)
      display.opacity = opacity

#------------------------------------------------------#

## 2 - Detect change - thresholding

# Kernel functions - Define erosion and dilation functions for convolving kernel through ratio
def erode_op(map_layer, iters, kernel):
    map_layer = ~map_layer
    for i in range(iters):
        map_layer = wf.conv2d(map_layer, kernel) > 0
    map_layer = ~map_layer 
    return map_layer

def dilate_op(map_layer, iters, kernel):
    for i in range(iters):
        map_layer = map_layer * 1.0
        map_layer = wf.conv2d(map_layer, kernel) > 0
    return map_layer

# Define function for plotting detected change
def plotChange(pv):
  # Detect change according to bounds for logarithmic ratio
  change = (pv['slider'].value[0] < pv['log_ratio']) & (pv['log_ratio'] < pv['slider'].value[1])

  # Apply dilatation to change
  eroded = erode_op(change, iters=1, kernel=pv['kernel'])
  dilated = dilate_op(eroded, iters=2, kernel=pv['kernel'])
  pv['detections'] = dilated

  # Visualize detections and apply mask for ocean or clouds
  if os.path.exists(pv['mask']):
      omit = gpd.read_file(pv['mask']) # Load coatlines
      omitMask = Feature(geometry=omit.geometry[0],properties={}).rasterize(value=1) # Mask sea
      detection = dilated.mask(dilated==0).mask(omitMask==1).visualize('Detected Change', checkerboard=False, colormap='plasma', map=pv['m2'])
      pv['omitMask'] = omitMask
  else: detection = dilated.mask(dilated==0).visualize('Detected Change', checkerboard=False, colormap='plasma', map=pv['m2'])
  detection.opacity = 0.7
  
  return pv 
  
# Load in damage geojson from Copernicus EMS data and plot
def plotDamages(v,m,bounds=[]):
  try: settlements = gpd.read_file(v['damageAssessment']).to_crs({'init': 'epsg:4326'})
  except: print("Damage file not found.")
  color_dict = {'Not Applicable':'green','Negligible to slight damage':'blue', 'Moderately Damaged':'yellow', 'Highly Damaged':'orange', 'Completely Destroyed':'red'}

  # Filter settlements to be within specified damage grade and location polygon
  damage = settlements[settlements.grading.isin(v['grades'])]
  fillOpacity = 0.2
  if bounds: 
    damage = damage[damage.within(bounds)]
    fillOpacity = 0.4

  geo_data = GeoData(geo_dataframe = damage,
                   style={'color': 'red', 'radius':2, 'fillColor': 'red', 'opacity':fillOpacity, 'weight':1.9, 'dashArray':'2', 'fillOpacity':fillOpacity},
                   hover_style={'fillColor': 'red' , 'fillOpacity': fillOpacity},
                   point_style={'radius': 3, 'color': 'red', 'fillOpacity': fillOpacity, 'fillColor': 'red', 'weight': 3},
                   name = 'Damages')
  
  m.add_layer(geo_data)
    
  return m
  

# Main thresholding function
def thresholding(v):
    # Build logarithmic ratio adding together values from each band
  for i in v['bands']: 
      globals()['log_ratio'+i] = wf.log10(globals()['mos_1'+i] / globals()['mos_2'+i]) # Ratio for band
      log_ratio = globals()['log_ratio'+i] if (i is v['bands'][0]) else log_ratio + globals()['log_ratio'+i] # log(a)+log(b) = log(axb)
  
  # Define a kernel and perform one erosion followed by two dilations
  kernel = wf.Kernel(dims=(v['kSize'],v['kSize']), data=np.ones([1,v['kSize']**2]).tolist()[0])
  #     dims=(3,3), data=[0.,1.,0.,1.,1.,1.,0.,1.,0.,]) # for cross shaped kernel

  # Create map with before/after images upon which to superimpose detected change
  m2 = wf.map.map
  m2.center, m2.zoom = (v['latitude'], v['longitude']), v['zoom']
  before, after, ratio = globals()['mos_1'+str(v['visual'])].visualize('Before', map=m2), globals()['mos_2'+str(v['visual'])].visualize('After', map=m2), log_ratio.visualize('Ratio',colormap='plasma' , map=m2)
  before.opacity, after.opacity, ratio.opacity = 0,0.7,0
  display(ipywidgets.HBox([wf.map]))

  # Create slider to adjust ratio bounds
  slider = FloatRangeSlider(value=[v['threshold'], v['cap']], min=-0.5, max=0.5, step=0.01, description='Filter ratios',disabled=False, continuous_update=True, orientation='horizontal', readout=True, readout_format='.2f')
  display(slider)
  print('Run box below to display updated detection result')

  # Plot detected change
  plotVars = {'slider':slider,'log_ratio':log_ratio,'mask':v['mask'],'kernel':kernel,'m2':m2}
  plotVars = plotChange(plotVars)

  if 'damageAssessment' in v: m2 = plotDamages(v, m2)
  
  if not 'thresholdLegend' in v: # Add legend if forming map for first time
    l2 = LegendControl({"Detected Change":"#FFFF00","EMS Damage Recorded":"#FF0000"}) if 'damageAssessment' in v else LegendControl({"Detected Change":"#FFFF00"})
    m2.add_control(l2)
    v['thresholdLegend']=True
  
  return plotVars, v


#-----------------------------------------------------------#

## 3 - Detect change - U-Net classifier

# Display map upon which to draw Polygon for analysis
def drawPolygon(v):
  r = 4*v['area'] if 'area' in v else 4*0.0004
  testPoly = ipyleaflet.Polygon(locations=[(v['latitude']-r, v['longitude']-r), (v['latitude']-r, v['longitude']+r), (v['latitude']+r, v['longitude']+r),(v['latitude']+r, v['longitude']-r)], color="blue", fill_opacity=0, transform=True)
  m3 = wf.interactive.MapApp()
  m3.center, m3.zoom = (v['latitude'], v['longitude']), v['zoom']+1
  m3.add_layer(testPoly)
  for i in range(len(v['st_date'])): getImage(v,i,v['visual'],m3,0.7)
  
  return m3, testPoly


# Functions retrieving desired tiles ratio image and Sentinel imagery for display
def get_ratio_image(dltile_key,ratio,tilesize,bands,v):
  tile = dl.scenes.DLTile.from_key(dltile_key)
  sc, ctx = dl.scenes.search(aoi=tile, products=v['satellite'], start_datetime=v['st_date'][0], end_datetime=v['end_date'][0])
  return ratio.compute(ctx).ndarray.reshape(tilesize,tilesize,len(bands)) 

def get_sentinel_image(dltile_key, bands,v):
  tile = dl.scenes.DLTile.from_key(dltile_key)
  sc, ctx = dl.scenes.search(aoi=tile, products=v['satellite'], start_datetime=v['st_date'][0], end_datetime=v['end_date'][0])
  im = sc.mosaic(bands=bands, ctx=ctx, bands_axis=-1)
  return im, ctx


# Function running predict image for each tile
def predict_image(dltile_key,ratio,tilesize,bands,v):
  # load model
  modelName = base+"models/optimalModel" # Change here if using own model
  model = load_model(modelName)
  # get imagery
  im = get_ratio_image(dltile_key,ratio,tilesize,bands,v)
  # add batch dimension
  im = np.expand_dims(im, axis=0).astype(np.float32)
  # predict
  pred = model.predict(im,verbose=0)

  return im, pred


## Function to get detections for each tile
def testTile(lat,lon,tilesize,threshold,v,ratio):
  tile = dl.scenes.DLTile.from_latlon(lat, lon, resolution=v['resolution'], tilesize=tilesize, pad=0) # Convert coordinates to nearest descartes labs tile with size of our choosing

  im, pred = predict_image(tile.key,ratio,tilesize,v['visual'],v) # Run prediction function for tile
  sent, ctx = get_sentinel_image(tile.key,v['visual'],v) # Get Sentinel imagery for tile

  disting = pred > threshold # Get damaged predictions

  # Extract latitude & longitude of each pixel in prediction (whether true or false)
  bounds, disting = ctx.bounds, disting[0,:,:,0] if len(disting.shape) == 4 else disting # Get bounds from tile and reduce extra dimensionality of classification matrix
  lats, longs = np.linspace(bounds[3],bounds[1],disting.shape[0]), np.linspace(bounds[0],bounds[2],disting.shape[1]) # Vector of lat, longs

  # Create matrix of coordinates for pixels with change detected
  xm, ym = np.meshgrid(longs,lats)
  xc, yc = xm*(disting), ym*(disting)

  # Get geodataframe for pixel points
  df = pd.DataFrame(columns=['Northing', 'Easting'])
  for i,j in zip(np.nonzero(xc)[0], np.nonzero(xc)[1]):
      df = df.append({'Northing': yc[i][j],'Easting': xc[i][j]}, ignore_index=True)
  det = gpd.GeoDataFrame(df, crs={'init':ctx.bounds_crs}, geometry=gpd.points_from_xy(df.Easting, df.Northing)).to_crs({'init': 'epsg:4326'})

  return det, ctx 

# Function looping through all tiles and assembling detections
def classifyDamage(testPoly, v, m3):
  ## Loop through tiles to get all detections
  tilesize = 16 # Optimal model tilesize used in demo
  # Get latitudes and longitudes for tiles according to polygon drawn and tilesize
  if type(testPoly.locations[0][0]) is float: # Default for polygon
    tileLats = np.arange(testPoly.locations[0][0],testPoly.locations[2][0],v['resolution']*1E-5*tilesize)
    tileLons = np.arange(testPoly.locations[0][1],testPoly.locations[2][1],v['resolution']*1E-5*tilesize)
  else: # If polygon has been modified in the map
    tileLats = np.arange(testPoly.locations[0][0]['lat'],testPoly.locations[0][2]['lat'],v['resolution']*1E-5*tilesize)
    tileLons = np.arange(testPoly.locations[0][0]['lng'],testPoly.locations[0][2]['lng'],v['resolution']*1E-5*tilesize)
  print("Number of tiles requested:",len(tileLats)*len(tileLons),". Approximately",8*len(tileLats)*len(tileLons),"seconds on 16GB RAM.")
  
  threshold = 0.5 # Threshold of likelihood for asserting change has occured
  
  # Calculate logarithmic ratio for RGB images
  for i in range(len(v['st_date'])): getImage(v,i,v['visual'],m3,0)
  ratio = wf.log10(globals()['mos_1'+str(v['visual'])] / globals()['mos_2'+str(v['visual'])])
  
  allDet = gpd.GeoDataFrame(crs={'init': 'epsg:4326'})
  allCtx = np.array([])
  for lat in tqdm(tileLats, desc='Latitude Rows'):
      for lon in tqdm(tileLons, desc='Longitude Columns'):
          newDet, newCtx = testTile(lat,lon,tilesize,threshold,v,ratio)
          newDet.index = newDet.index + len(allDet.index)
          allDet = allDet.append(newDet)
          allCtx = np.append([allCtx], [np.array(newCtx.bounds)])
  
  getImage(v,1,v['visual'],m3,0.7) # Display sentinel imagery using function from map 1
  
  # Assessment layer for detections from model
  allDet_data = GeoData(geo_dataframe = allDet, 
                        style={'color': 'yellow', 'radius':2, 'fillColor': 'yellow', 'opacity':0.7, 'weight':1.9, 'dashArray':'2', 'fillOpacity':0.7},
                        point_style={'radius': 2, 'color': 'yellow', 'fillOpacity': 0.7, 'fillColor': 'blue', 'weight': 3},
                        name = 'Detected Change')
  m3.add_layer(allDet_data)
  
  if 'damageAssessment' in v:
    bounds = gpd.GeoSeries(Polygon.from_bounds(min(allCtx[0::4]),min(allCtx[1::4]),max(allCtx[2::4]),max(allCtx[3::4])),
                         crs={'init':newCtx.bounds_crs}).to_crs(epsg=4326).geometry[0]
    m3 = plotDamages(v, m3, bounds)

  if not 'classifyLegend' in v: # Add legend if forming map for first time
    l3 = LegendControl({"Detected Change":"#FFFF00", "Damage Recorded":"#FF0000", "Search Area":"#0000FF"}) if 'damageAssessment' in v else LegendControl({"Detected Change":"#FFFF00", "Search Area":"#0000FF"})
    m3.add_control(l3)
    v['classifyLegend']=True
    
  return {'allDetections':allDet,'plotDetections':allDet_data,'allCtx':allCtx,'finalCtx':newCtx}, v
    
#--------------------------------------------------------------# 

## 4.1 Thresholding Accuracy
  
def thresholdingAccuracy(pv,v):
  # Convert mask into coordinate array
  # Get vector of pixels which have changed coordinates
  gtx, detection = wf.map.geocontext(), pv['detections'].mask(pv['detections']==0).mask(pv['omitMask']==1) if os.path.exists(pv['mask']) else pv['detections'].mask(pv['detections']==0)
  print('Extracting detections from image for evaluation.')
  change = detection.compute(geoctx=gtx)

  # Get latitude & longitude of each pixel in mask (whether true or false)
  bounds = change.geocontext['bounds']
  lats, longs = np.linspace(bounds[3],bounds[1],change.geocontext['arr_shape'][0]), np.linspace(bounds[0],bounds[2],change.geocontext['arr_shape'][1])

  # Create matrix of coordinates for pixels with change detected
  xm, ym = np.meshgrid(longs,lats)
  xc, yc = xm*(1-change.ndarray.mask[0]), ym*(1-change.ndarray.mask[0])

  # Get geodataframe for pixel points
  df = pd.DataFrame(columns=['Latitude', 'Longitude'])
  for i,j in tqdm(zip(np.nonzero(xc)[0], np.nonzero(xc)[1])):
      df = df.append({'Latitude': yc[i][j],'Longitude': xc[i][j]}, ignore_index=True)

  det = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))
  
  # Load in damage geojson from Copernicus EMS data
  try: 
      settlements = gpd.read_file(v['damageAssessment']).to_crs({'init': 'epsg:4326'})
      color_dict = {'Not Applicable':'green','Negligible to slight damage':'blue', 'Moderately Damaged':'yellow', 'Highly Damaged':'orange', 'Completely Destroyed':'red'}

      # Filter settlements to be within specified damage grade and location polygon
      damage = settlements[settlements.grading.isin(v['grades'])]
      damage = damage[damage.within(Polygon(change.geocontext['geometry']['coordinates'][0]))]

      # Filter detections to area covered by damage inspection
      try: det = det.loc[(det.geometry.x < max(damage.geometry.x)) & (det.geometry.x > min(damage.geometry.x)) & (det.geometry.y < max(damage.geometry.y)) & (det.geometry.y > min(damage.geometry.y))]
      except: pass
  except: print("No damage file set for this area. Please choose a location with an EMS damage file")
          
  # Create polygons around point locations of damages
  if not os.path.exists(v['damageGeojson']) and damage.geometry[damage.index[0]].type is not 'Polygon': # Gets point assessment damages into geojson file
    print('Damage geojson not found - Recreating damage polygons from EMS assessment')
    features = []
    for i in tqdm(damage.index):
        poly = Polygon([[damage.geometry.x[i], damage.geometry.y[i]],
                        [damage.geometry.x[i]+v['area'],damage.geometry.y[i]],
                        [damage.geometry.x[i]+v['area'], damage.geometry.y[i]+v['area']],
                        [damage.geometry.x[i], damage.geometry.y[i]+v['area']],
                        [damage.geometry.x[i], damage.geometry.y[i]]])
        features.append(geojson.Feature(properties={"Damage": damage.grading[i]}, geometry=poly))

    fc = geojson.FeatureCollection(features)
    with open(v['damageGeojson'], 'w') as f: geojson.dump(fc, f)

  elif not os.path.exists(v['damageGeojson']):  # Puts polygon assessments into geojson file
    with open(v['damageGeojson'], 'w') as f: geojson.dump(damage, f)
        
  # Load building damages
  dmg = gpd.read_file(v['damageGeojson'])
  print('Changed pixels:',len(det), '\nDamaged buildings:',len(dmg))

  # Initialise accuracy and recall vectors
  acc, rec = np.zeros([max(dmg.index)+1,1]), np.zeros([max(det.index)+1,1]) # Initialise accuracy, recall arrays

  # Loop through pixels to determine recall (if pixel corresponds to damaged building)
  for i in tqdm(det.index,desc='Pixels evaluated'):
      # Loop through building to determine accuracy (damaged building has been detected)
      for j in dmg.index:
          if det.geometry[i].within(dmg.geometry[j]):
              rec[i,0], acc[j,0] = True, True

  # Calculate metrics from vector outputs
  a = sum(acc)/len(dmg)
  r = sum(rec)/len(det)
  f1 = 2*(a*r)/(a+r)
  print('Precision:',a[0],'\nRecall:',r[0],'\nF1 score:',f1[0])
  
  # Damage detected true/false
  dmg['found'] = pd.Series(acc[:,0], index=dmg.index)
  
  ## Display on interactive map
  # Initialise map
  m4 = wf.interactive.MapApp()
  m4.center, m4.zoom = (v['latitude'], v['longitude']), v['zoom']

  # Plot background imagery as image 2 using function from map 1
  getImage(v,1,v['visual'],m4,0.7,visualise=True)

  # Add layers for building polygons whether red for not found, green for found
  not_found = GeoData(geo_dataframe = dmg.loc[dmg['found']==0], style={'color': 'red', 'radius':2, 'fillColor': 'red', 'opacity':0.7, 'weight':1.9, 'dashArray':'2', 'fillOpacity':0.7},
                      hover_style={'fillColor': 'red' , 'fillOpacity': 0.5},
                      name = 'Damages')
  found = GeoData(geo_dataframe = dmg.loc[dmg['found']==1], style={'color': 'green', 'radius':2, 'fillColor': 'green', 'opacity':0.7, 'weight':1.9, 'dashArray':'2', 'fillOpacity':0.7},
                      hover_style={'fillColor': 'green' , 'fillOpacity': 0.5},
                      name = 'Damages')
  m4.add_layer(not_found)
  m4.add_layer(found)

  # Plot pixels where change has been detected
  try: detection = pv['detections'].mask(pv['detections']==0).mask(pv['omitMask']==1).visualize('Detected Change', checkerboard=False, colormap='plasma', map=m4)
  except: detection = pv['detections'].mask(pv['detections']==0).visualize('Detected Change', checkerboard=False, colormap='plasma', map=m4)
  detection.opacity = 0.7

  # Legend
  m4.add_control(LegendControl({"Detected Change":"#FFFF00", "Damage Identified":"#008000", "Damage Not Identified":"#FF0000"})) 

  return m4

#-----------------------------------#

## 4.2 - Classifier Accuracy

def classifierAccuracy(cl, v):
# Load building damages and filter for within detection area
  allDet, allDet_data, allCtx = cl['allDetections'],cl['plotDetections'],cl['allCtx']
  
  try: dmg = gpd.read_file(v['damageGeojson'])
  except: print("No damage file set for this area. Please choose a location with an EMS damage file")
  filtered = gpd.GeoDataFrame(crs={'init': 'epsg:4326'})

  tilePoly = gpd.GeoSeries(Polygon.from_bounds(min(allCtx[0::4]),min(allCtx[1::4]),max(allCtx[2::4]),max(allCtx[3::4])), crs={'init':cl['finalCtx'].bounds_crs}).to_crs(epsg=4326).geometry[0]
  for i in dmg.index: 
      if dmg.geometry[i].centroid.within(tilePoly):
          filtered = filtered.append(dmg.loc[i])

  print('Changed pixels:',len(allDet), '\nDamaged buildings:',len(filtered))

  # Initialise accuracy and recall vectors
  acc, rec = np.zeros([max(filtered.index)+1,1]), np.zeros([max(allDet.index)+1,1]) # Initialise accuracy, recall arrays

  # Loop through pixels to determine recall (if pixel corresponds to damaged building)
  for i in tqdm(allDet.index):
      # Loop through building to determine accuracy (damaged building has been detected)
      for j in filtered.index:
          if allDet.geometry[i].within(filtered.geometry[j]):
              rec[i,0], acc[j,0] = True, True

  # Calculate metrics from vector outputs
  a = sum(acc)/len(filtered)
  r = sum(rec)/len(allDet)
  f1 = 2*(a*r)/(a+r)
  print('Accuracy:',a[0],'\nRecall:',r[0],'\nF1 score:',f1[0]) 
  
  # Initialise map
  m5 = wf.interactive.MapApp()
  m5.center, m5.zoom = (v['latitude'], v['longitude']), v['zoom']

  getImage(v,1,v['visual'],m5,0.7,visualise=True) # Display sentinel imagery using function from map 1

  # Ass layer for detections from model
  m5.add_layer(allDet_data)

  # Add layers for building polygons whether red for not found, green for found
  filtered['found'] = pd.Series(acc[filtered.index,0], index=filtered.index)
  all_not_found = GeoData(geo_dataframe = filtered.loc[filtered['found']==0], style={'color': 'red', 'radius':2, 'fillColor': 'red', 'opacity':0.7, 'weight':1.9, 'dashArray':'2', 'fillOpacity':0.7},
                      hover_style={'fillColor': 'red' , 'fillOpacity': 0.5},
                      name = 'Damages')
  all_found = GeoData(geo_dataframe = filtered.loc[filtered['found']==1], style={'color': 'green', 'radius':2, 'fillColor': 'green', 'opacity':0.7, 'weight':1.9, 'dashArray':'2', 'fillOpacity':0.7},
                      hover_style={'fillColor': 'green' , 'fillOpacity': 0.5},
                      name = 'Damages')
  m5.add_layer(all_not_found)
  m5.add_layer(all_found)

  # Legend
  m5.add_control(LegendControl({"Damage Identified":"#008000", "Damage Not Identified":"#FF0000", "Detected Change":"#FFFF00", "Search Area":"#0000FF"}))

  # Plot bounding box for damage search
  tb = tilePoly.bounds
  evalArea = ipyleaflet.Polygon(locations=[(tb[1],tb[0]), (tb[1],tb[2]), (tb[3],tb[2]),(tb[3],tb[0])],
                                color="blue", weight=2, fill_opacity=0, transform=False)
  m5.add_layer(evalArea)

  return m5


#--------------END------------------#