#!/bin/bash

pip install gdown

pwd='../change_detection/ratio_method/download_data'

# results
gdown -O $(pwd)/results.zip "https://drive.google.com/uc?export=download&id=18AngZi11uDFjNRYlM0kSdUPRB5Q0O8_v"
unzip $(pwd)/results.zip
rm $(pwd)/results.zip

# records
gdown -O $(pwd)/records.zip "https://drive.google.com/uc?export=download&id=1PoXkx9sf5Aiw6HYIetTGFTrgy_JPOWb0"
unzip $(pwd)/records.zip
rm $(pwd)/records.zip

# models
gdown -O $(pwd)/models.zip "https://drive.google.com/uc?export=download&id=18wdEMvBKjXdM8zB2kD9VRU2KO3v4ouBR"
unzip $(pwd)/models.zip
rm $(pwd)/models.zip

# gradings
gdown -O $(pwd)/gradings.zip "https://drive.google.com/uc?export=download&id=14gH7lV6nGoaN4GQjzIwFVDT84J6QEMiT"
unzip $(pwd)/gradings.zip
rm $(pwd)/gradings.zip

# geojsons
gdown -O $(pwd)/geojsons.zip "https://drive.google.com/uc?export=download&id=1ZDzQKo37f51ZC2uRVjT8YNSusaNODRol"
unzip $(pwd)/geojsons.zip
rm $(pwd)/geojsons.zip

# coastlines
gdown -O $(pwd)/coastlines.zip "https://drive.google.com/uc?export=download&id=1fx_WeWbRmSn9gWC8fuWAIUksaIQLay6u"
unzip $(pwd)/coastlines.zip
rm $(pwd)/coastlines.zip