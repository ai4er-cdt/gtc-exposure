#!/bin/bash

#labelled tifs
wget -O ucm.zip "http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip"
unzip ucm.zip
mv UCMerced_LandUse/Images data/Images

rm ucm.zip
rm -r UCMerced_LandUse
