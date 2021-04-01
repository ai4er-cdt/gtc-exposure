#!/bin/bash

### Settlement Segmentation ###
# Random Forrest Classifier
sh ./gtc-exposure/settlement_segmentation/rrandomforest/download-demo-data.sh

### Change Detection ###
# Ratio Method
sh ./gtc-exposure/change_detection/ratio_method/download_data.sh


### Exposure Quantification ###
sh ./gtc-exposure/exposure_quantification/download-data.sh