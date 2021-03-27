### Deep change detection

This directory contains code for a deep learning approach to change detection on Sentinel-2 data. To start training your own model, download the OSCD dataset (https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection) and run cd_dl_notebook, which will allow you to train and test the U-Net model on the dataset. This notebook was adapted from the work of Daudt [1]. If you wish to use a pre-trained model, the weights for the U-Net model are available in the weights directory. This trained model achieved an F1-score of 0.45 on the OSCD test set.

To experiment with training the model further, possible loss functions are provided in losses.py, including Lovasz and Dice loss functions.

The notebook sentinel_cd_pipeline contains code to run the trained model on test locations, using the Descartes Labs platform, while the folder demo contains .py files containing the functions for the demo, which draws on the sentinel_cd_pipeline notebook. A full demo will be available at https://www.bas.ac.uk/project/ai4eoaccelerator/exposure/ shortly.

Example images (including results of the change detection algorithm) can be found in example_images.

[1] Daudt, R.C., Le Saux, B., Boulch, A. and Gousseau, Y., 2018, July. Urban change detection for multispectral earth observation using convolutional neural networks. In IGARSS 2018-2018 IEEE International Geoscience and Remote Sensing Symposium (pp. 2115-2118). IEEE.
