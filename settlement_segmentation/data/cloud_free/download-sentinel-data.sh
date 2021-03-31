#!/bin/bash

pip install gdown

#unlabelled tifs
gdown -O train_tifs.zip "https://drive.google.com/uc?export=download&id=15gnDamDgZA_ypq4Pj0040FAd-ZXtlpRy"
unzip train_tifs.zip
mkdir data
mv tif data/sentinel_unlabelled
rm train_tifs.zip

#labelled tifs
wget -O test_tifs.zip "https://drive.google.com/uc?export=download&id=1J-T4Xvz92pVFTgdyuAo_92Eq2vTHGZmX"
unzip test_tifs.zip
mv tif data/sentinel_labelled
rm test_tifs.zip

