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
#%matplotlib inline

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
from change_detection.ratio_method.unet import UNet

#-------------------------------------------------------#

base = '/change_detection/ratio_method/'

## Assigning locations and variables

def chooseLocation(): 
  return ipywidgets.Dropdown(placeholder='Choose Location', 
                                 options=[('train - Roseau, Dominica',0), ('train - Abricots, Haiti',1), ('test - Jeremie, Haiti',2), ('test - Port Salut, Haiti',3), ('settlement - Bidonville KSV, Haiti',4), ('settlement - Parry Town, Jamaica',5), ('high resolution - Hidalgo, Texas',6),('New Location',7)],
                                 description='Location:',
                                 disabled=False)

def assignVariables(locationFunc):
  location = locationFunc.value

  if location == 7:
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
                                 options=[('red',['red']), ('green',['green']), ('blue',['blue']), ('red,green,blue',['red','green','blue']), ('near IR',['nir'])],
                                 description='Imagery bands:',
                                 disabled=False)
    newRes = ipywidgets.IntText(value=10, description='Satellite resolution:', disabled=False)
    newMask = ipywidgets.Text(value="", placeholder='mask.geojson', description='Path to mask if using one:', disabled=False)
    display(newLat,newLon,newZoom,newStart1,newEnd1,newStart2,newEnd2,newSat,
            newCloud,newThreshold,newCap,newBand,newRes,newMask)

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
    allCaps = [0.3,0.1,0.1,0.1,0.3,0.3,-0.1] # Initial bounds for logarithmic ratio change detection
    allBands = [['red'],['green'],['red'],['red','green','blue'],['red'],['red'],['red','green','blue']] # Bands used to evaluate ratio
    resolution = [10,10,10,10,10,10,1]
    satellite =  "usda:naip:rgbn:v1" if location is 6 else "sentinel-2:L1C"

    # Assign to dictionary
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
      allDmgAssess = [base+"gradings/EMSR246_04ROSEAU_02GRADING_v1_5500_settlements_point_grading.dbf", base+"gradings/EMSR185_35ABRICOTS_02GRADING_v1_2500_settlements_point_grading.dbf", base+"gradings/EMSR185_11JEREMIE_02GRADING_MONIT01_v1_4000_settlements_point_grading.dbf", base+"gradings/EMSR185_09PORTSALUT_02GRADING_v1_5500_settlements_point_grading.dbf"] # Damage assessment database location (.dbf file needs .prj,.shp,.shx in same
      allDmgFiles = [base+'geojsons/'+'RoseauDamage0004g3.geojson',base+'geojsons/'+'HaitiAbricotsDamage0004g3.geojson',base+'geojsons/'+'HaitiJeremieDamage0004g3.geojson',base+'geojsons/'+'HaitiPortSalutDamage0004g3.geojson'] # Damage geojson with building footprints
      allMaskPoly = [base+'coastlines/swDominicaOcean.geojson', base+'coastlines/swHaitiCoastline.geojson', base+'coastlines/swHaitiCoastline.geojson',base+'coastlines/PortSalutCoastline.geojson'] # Geojson file masking feature such as ocean

      # Assign to dictionary
      variables['damageAssessment'], variables['damageGeojson'], variables['mask'] = allDmgAssess[location], allDmgFiles[location], allMaskPoly[location]
      variables['grades'] = ['Completely Destroyed','Highly Damaged','Moderately Damaged'] # Options: 'Not Applicable','Negligible to slight damage', 'Moderately Damaged', 'Highly Damaged', 'Completely Destroyed'
      variables['area'] = 0.0004 # Building polygon size in lat/long degrees
    
    button = ipywidgets.Button(description="Show variables")
    output = ipywidgets.Output()
    display(button, output)
    def on_button_clicked(b):
        with output:
            print(variables)
    button.on_click(on_button_clicked)
    
  return variables

def submitNewLocation(v):
  if not type(v['latitude']) == float:
    variables = {'latitude':v['latitude'].value,
                 'longitude':v['longitude'].value,
                 'zoom':v['zoom'].value,
                 'st_date':[v['st_date'][0].value, v['st_date'][1].value],
                 'end_date':[v['end_date'][0].value, v['end_date'][1].value],
                 'satellite':v['satellite'].value, # Descartes product name
                 'visual':['red','green','blue'], # Imagery bands for display purposes
                 'cloudFraction':v['cloudFraction'].value,
                 'threshold':v['threshold'].value,
                 'cap':v['cap'].value,
                 'bands':v['bands'][0].value,
                 'kSize':1, # Pixel dilatation kernel size
                 'resolution':v['resolution'].value,
                 'mask':v['mask'].value
                }
  else: variables = v
  return variables

#---------------------------------------------------#

## Plot satellite imagery

# Define function which retrieves imagery for specified time period and band. Allow specification of layer opacity and map number
def beforeAfterImages(variables):
  m1 = wf.interactive.MapApp()
  m1.center, m1.zoom = (variables['latitude'], variables['longitude']), variables['zoom']
  
  # Loop over dates (time 1 & ime 2)
  for i in range(len(variables['st_date'])):
      for j in variables['bands']: getImage(variables,i,j,m1,0,False) # Retrieve layer for each band to be used for ratio
      getImage(variables,i,variables['visual'],m1,1) # Retrieve layer for visual
  
  return m1

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

## Detect change through thresholding

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

  # Visualize detections and apply mask for ocean or clouds
  if os.path.exists(pv['mask']):
      omit = gpd.read_file(pv['mask']) # Load coatlines
      omitMask = Feature(geometry=omit.geometry[0],properties={}).rasterize(value=1) # Mask sea
      detection = dilated.mask(dilated==0).mask(omitMask==1).visualize('Detected Change', checkerboard=False, colormap='plasma', map=pv['m2'])
  else: detection = dilated.mask(dilated==0).visualize('Detected Change', checkerboard=False, colormap='plasma', map=pv['m2'])
  
  
# Load in damage geojson from Copernicus EMS data and plot
def plotDamages(v,m,bounds=[]):
  try: settlements = gpd.read_file(os.getcwd()+v['damageAssessment']).to_crs({'init': 'epsg:4326'})
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
  plotChange({'slider':slider,'log_ratio':log_ratio,'mask':v['mask'],'kernel':kernel,'m2':m2})

  
  if 'damageAssessment' in v: m2 = plotDamages(v, m2)
  
  if not 'l2' in globals(): # Add legend if forming map for first time
      l2 = LegendControl({"Detected Change":"#FFFF00","Damage Recorded":"#FF0000"}) if 'damageAssessment' in v else LegendControl({"Detected Change":"#FFFF00"})
      m2.add_control(l2)
  
  return {'slider':slider,'log_ratio':log_ratio,'mask':v['mask'],'kernel':kernel,'m2':m2}


#-----------------------------------------------------------#

## Detect change through classifier

# Display map upon which to draw Polygon for analysis
def drawPolygon(v):
  r = 4*v['area'] if 'area' in v else 4*0.0004
  testPoly = ipyleaflet.Polygon(locations=[(v['latitude']-r, v['longitude']-r), (v['latitude']-r, v['longitude']+r), (v['latitude']+r, v['longitude']+r),(v['latitude']+r, v['longitude']-r)], color="yellow", fill_color="yellow", transform=True)
  m3 = wf.interactive.MapApp()
  m3.center, m3.zoom = (v['latitude'], v['longitude']), v['zoom']+1
  #pos = Map(center=(v['latitude'], v['longitude']), zoom=v['zoom']+1)
  testPoly.color, testPoly.fill_opacity = 'blue', 0
  m3.add_layer(testPoly)
  for i in range(len(v['st_date'])): getImage(v,i,v['visual'],m3,0.7)
  
  return m3, testPoly
  
def get_ratio_image(dltile_key,ratio,tilesize,bands,v):
  tile = dl.scenes.DLTile.from_key(dltile_key)
  sc, ctx = dl.scenes.search(aoi=tile, products=v['satellite'], start_datetime=v['st_date'][0], end_datetime=v['end_date'][0])
  return ratio.compute(ctx).ndarray.reshape(tilesize,tilesize,len(bands)) 

# Function retrieving desired tile from Sentinel imagery for display
def get_sentinel_image(dltile_key, bands,v):
  tile = dl.scenes.DLTile.from_key(dltile_key)
  sc, ctx = dl.scenes.search(aoi=tile, products=v['satellite'], start_datetime=v['st_date'][0], end_datetime=v['end_date'][0])
  im = sc.mosaic(bands=bands, ctx=ctx, bands_axis=-1)
  return im, ctx

# Function running predict image for each tile
def predict_image(dltile_key,ratio,tilesize,bands,v):
  #print("Predict on image for dltile {}".format(dltile_key))

  # load model
  modelName = os.getcwd()+base+"models/optimalModel"
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

def classifyDamage(testPoly, v, m3):
  ## Loop through tiles to get all detections
  tilesize = 16 # Optimal model tilesize used in demo
  # Get latitudes and longitudes for tiles according to polygon drawn and tilesize
  tileLats = np.arange(testPoly.locations[0][0]['lat'],testPoly.locations[0][2]['lat'],v['resolution']*1E-5*tilesize)
  tileLons = np.arange(testPoly.locations[0][0]['lng'],testPoly.locations[0][2]['lng'],v['resolution']*1E-5*tilesize)
  print("Number of tiles requested:",len(tileLats)*len(tileLons),". Approximately",8*len(tileLats)*len(tileLons),"seconds on 16GB RAM.")
  threshold = 0.5
  
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
  
  # Ass layer for detections from model
  allDet_data = GeoData(geo_dataframe = allDet, 
                        style={'color': 'yellow', 'radius':2, 'fillColor': 'yellow', 'opacity':0.7, 'weight':1.9, 'dashArray':'2', 'fillOpacity':0.7},
                        point_style={'radius': 2, 'color': 'yellow', 'fillOpacity': 0.7, 'fillColor': 'blue', 'weight': 3},
                        name = 'Detected Change')
  m3.add_layer(allDet_data)
  
  if 'damageAssessment' in v:
    bounds = gpd.GeoSeries(Polygon.from_bounds(min(allCtx[0::4]),min(allCtx[1::4]),max(allCtx[2::4]),max(allCtx[3::4])),
                         crs={'init':newCtx.bounds_crs}).to_crs(epsg=4326).geometry[0]
    m3 = plotDamages(v, m3, bounds)
    
  if not 'l3' in globals(): # Add legend if forming map for first time
    l3 = LegendControl({"Detected Change":"#FFFF00", "Damage Recorded":"#FF0000", "Search Area":"#0000FF"}) if 'damageAssessment' in v else LegendControl({"Detected Change":"#FFFF00", "Search Area":"#0000FF"})
    m3.add_control(l3)
    
  
  
  
          
  