### Deep change detection

This directory contains code for the deep learning approach to change detection on Sentinel-2 data. cd_dl_notebook
contains the code used to train and test the U-Net model on the OSCD dataset. 

Weights for the trained U-Net model are available in the weights directory.

Additional loss functions are provided in losses.py.

Example images (including results of the change detection algorithm) can be found in example_images.

The notebook sentinel_cd_pipeline contains code to run the trained model on test locations, using the Descartes Labs platform.
