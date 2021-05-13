#!/bin/bash

pwd='../settlement_segmentation/randomforest'

wget -O $(pwd)/classifier_data/sentinel_raster_maps/image_0.tiff "https://drive.google.com/uc?export=download&id=1fZ_0zMfVofB6kVmZacUC6nGPNKPNZX-P"
wget -O $(pwd)/classifier_data/target_raster_maps/target_0.tiff "https://drive.google.com/uc?export=download&id=1Bv5BpT7cV1QwfspOJDZOyq7dqAu-IE4R"
