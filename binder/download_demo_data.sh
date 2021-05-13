#!/bin/bash

### Settlement Segmentation ###
# Random Forrest Classifier
chmod +x ../settlement_segmentation/randomforest/download-demo-data.sh
sh ./../settlement_segmentation/randomforest/download-demo-data.sh

### Change Detection ###
# Ratio Method
chmod +x ../change_detection/ratio_method/download_dat
sh ./../change_detection/ratio_method/download_data.sh


### Exposure Quantification ###
chmod +x ../exposure_quantification/download-data.sh
sh ./../exposure_quantification/download-data.sh
