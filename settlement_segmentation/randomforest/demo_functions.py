import numpy as np
import os
import tifffile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import pandas
import pylab as pl
import pickle 
import ipywidgets

from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer, jaccard_score

def classify(variables, model_loc):
    if not(variables['latitude']== 18.5113 and variables['longitude'] == -72.2974):
        print('Demo functionality only implemented for Bidonville')
        return
    else:
        model = pickle.load(open(model_loc, 'rb'))
        X_raster = tifffile.imread('settlement_segmentation/randomforest/killick_val.tiff')
        X_data = X_raster.reshape((X_raster.shape[0]*X_raster.shape[1], X_raster.shape[2]), order='F')
        
        y_pred = model.predict(X_data)
        y_pred_raster = y_pred.reshape((X_raster.shape[0], X_raster.shape[1]), order='F')
        
        resultsFig, resultsAx = plt.subplots(1,2, figsize = (30,30))
        resultsAx[0].imshow(X_raster)
        resultsAx[0].title.set_text('Sentinel RGB')
        resultsAx[1].imshow(y_pred_raster)
        resultsAx[1].title.set_text('Target')
        
        return
