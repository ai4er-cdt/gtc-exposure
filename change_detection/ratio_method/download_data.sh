#!/bin/bash

pip install gdown

mkdir change_detection/ratio_method/download_data/

# results
gdown -O results.zip "https://drive.google.com/uc?export=download&id=18AngZi11uDFjNRYlM0kSdUPRB5Q0O8_v"
unzip results.zip
rm results.zip
mv results change_detection/ratio_method/download_data/results

# records
gdown -O records.zip "https://drive.google.com/uc?export=download&id=1PoXkx9sf5Aiw6HYIetTGFTrgy_JPOWb0"
unzip change_detection/ratio_method/download_data/records.zip
rm records.zip
mv records change_detection/ratio_method/download_data/records

# models
gdown -O models.zip "https://drive.google.com/uc?export=download&id=18wdEMvBKjXdM8zB2kD9VRU2KO3v4ouBR"
unzip models.zip
rm models.zip
mv models change_detection/ratio_method/download_data/models

# gradings
gdown -O gradings.zip "https://drive.google.com/uc?export=download&id=14gH7lV6nGoaN4GQjzIwFVDT84J6QEMiT"
unzip gradings.zip
rm gradings.zip
mv gradings change_detection/ratio_method/download_data/gradings

# geojsons
gdown -O geojsons.zip "https://drive.google.com/uc?export=download&id=1ZDzQKo37f51ZC2uRVjT8YNSusaNODRol"
unzip geojsons.zip
rm geojsons.zip
mv geojsons change_detection/ratio_method/download_data/geojsons

# coastlines
gdown -O coastlines.zip "https://drive.google.com/uc?export=download&id=1fx_WeWbRmSn9gWC8fuWAIUksaIQLay6u"
unzip coastlines.zip
rm coastlines.zip
mv coastlines change_detection/ratio_method/download_data/coastlines