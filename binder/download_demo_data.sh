#!/bin/bash

### Settlement Segmentation ###
# Random Forrest Classifier
sh ./settlement_segmentation/rrandomforest/download-demo-data.sh

### Change Detection ###
# Ratio Method
sh ./change_detection/ratio_method/download_data.sh


### Exposure Quantification ###
sh ./exposure_quantification/download-data.sh