#!/bin/bash

### Settlement Segmentation ###

# Random Forrest Classifier
sh ./../settlement_segmentation/randomforest/download-data.sh

# Unclassified Learning - UCM Land Use Data
sh ./../settlement_segmentation/liunsupervised/download-ucm-data.sh

# SUnclassified Learning - Cloud Free Sentinel Data
sh ./../settlement_segmentation/data/cloud_free/download-sentinel-data.sh


### Change Detection ###

# Ratio Method
sh ./../change_detection/ratio_method/download_data.sh

# XBD High Resolution
sh ./../change_detection/xbd_hi_resolution_attempt/download-data.sh


### Exposure Quantification ###

sh ./../exposure_quantification/download-data.sh