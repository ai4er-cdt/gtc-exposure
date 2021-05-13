#!/bin/bash

pip install gdown

mkdir change_detection/ratio_method/download_data/

# results
gdown -O change_detection/ratio_method/download_data/results.zip "https://drive.google.com/uc?export=download&id=18AngZi11uDFjNRYlM0kSdUPRB5Q0O8_v"
unzip change_detection/ratio_method/download_data/results.zip
rm change_detection/ratio_method/download_data/results.zip

# records
gdown -O change_detection/ratio_method/download_data/records.zip "https://drive.google.com/uc?export=download&id=1PoXkx9sf5Aiw6HYIetTGFTrgy_JPOWb0"
unzip change_detection/ratio_method/download_data/records.zip
rm change_detection/ratio_method/download_data/records.zip

# models
gdown -O change_detection/ratio_method/download_data/models.zip "https://drive.google.com/uc?export=download&id=18wdEMvBKjXdM8zB2kD9VRU2KO3v4ouBR"
unzip change_detection/ratio_method/download_data/models.zip
rm change_detection/ratio_method/download_data/models.zip

# gradings
gdown -O change_detection/ratio_method/download_data/gradings.zip "https://drive.google.com/uc?export=download&id=14gH7lV6nGoaN4GQjzIwFVDT84J6QEMiT"
unzip change_detection/ratio_method/download_data/gradings.zip
rm change_detection/ratio_method/download_data/gradings.zip

# geojsons
gdown -O change_detection/ratio_method/download_data/geojsons.zip "https://drive.google.com/uc?export=download&id=1ZDzQKo37f51ZC2uRVjT8YNSusaNODRol"
unzip change_detection/ratio_method/download_data/geojsons.zip
rm change_detection/ratio_method/download_data/geojsons.zip

# coastlines
gdown -O change_detection/ratio_method/download_data/coastlines.zip "https://drive.google.com/uc?export=download&id=1fx_WeWbRmSn9gWC8fuWAIUksaIQLay6u"
unzip change_detection/ratio_method/download_data/coastlines.zip
rm change_detection/ratio_method/download_data/coastlines.zip