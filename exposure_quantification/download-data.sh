#!/bin/bash

pwd='../exposure_quantification'

wget -O $(pwd)/GHS_png.zip "https://drive.google.com/uc?export=download&id=1WTohi_XZrJXyHzKVueKee-S3QJLtMocK"
unzip $(pwd)/GHS_png.zip

rm $(pwd)/GHS_png.zip
