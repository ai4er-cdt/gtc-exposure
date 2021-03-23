#!/bin/bash

wget -O test_data.zip "https://drive.google.com/uc?export=download&id=1KpUhuGNnL5rZWIDTnooYU1K2rDzQffUJ"
unzip test_data.zip

mv test_data inference/test_data
mv test_data_02 inference/test_data_02
mv test_data_03 inference/test_data_03
mv test_data_04 inference/test_data_04

rm test_data.zip
