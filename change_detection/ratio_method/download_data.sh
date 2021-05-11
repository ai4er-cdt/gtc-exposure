#!/bin/bash

pip install gdown

# results
gdown -O results.zip "https://drive.google.com/uc?export=download&id=18AngZi11uDFjNRYlM0kSdUPRB5Q0O8_v"
unzip results.zip
rm results.zip

# records
gdown -O records.zip "https://drive.google.com/uc?export=download&id=1PoXkx9sf5Aiw6HYIetTGFTrgy_JPOWb0"
unzip records.zip
rm records.zip

# models
gdown -O models.zip "https://drive.google.com/uc?export=download&id=18wdEMvBKjXdM8zB2kD9VRU2KO3v4ouBR"
unzip models.zip
rm models.zip

# gradings
gdown -O gradings.zip "https://drive.google.com/uc?export=download&id=14gH7lV6nGoaN4GQjzIwFVDT84J6QEMiT"
unzip gradings.zip
rm gradings.zip

# geojsons
gdown -O geojsons.zip "https://drive.google.com/uc?export=download&id=1ZDzQKo37f51ZC2uRVjT8YNSusaNODRol"
unzip geojsons.zip
rm geojsons.zip

# coastlines
gdown -O coastlines.zip "https://drive.google.com/uc?export=download&id=1fx_WeWbRmSn9gWC8fuWAIUksaIQLay6u"
unzip coastlines.zip
rm coastlines.zip
