This directory corresponds to the random forest classification model, which was inspired by [Mapping Informal Settlements in Developing Countries using Machine Learning and Low Resolution Multi-spectral Data](https://arxiv.org/abs/1901.00861).

* ```cloud_mosaic_test.ipynb``` contains experiments removing clouds from sentinel imagery.

* ```create_geojson.ipynb``` creates geoJSONs from the OSM building layer.

* ```demo_functions.py``` contains functions used in the demo notebook.

* ```kfold_random_forest.ipynb``` contains code for the random forest model and k-fold cross validation.

* ```tiff_generator.ipynb``` generates .tiff files for binary classification from satellite imagery based on input geoJSONs.

* ```utils_santo.py``` contains utility functions for ```tiff_generator.ipynb```.

* ```download-data.sh``` and ```download-demo-data.sh``` downloads the relevant .tiff files for the classifier/demo.
