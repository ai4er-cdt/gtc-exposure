Directory for code for the model found in [Unsupervised Multilayer Feature Learning for Satellite Image Scene Classification](https://ieeexplore.ieee.org/document/7374663/). Code is original (i.e written by Matt rather than the author of the paper, who doesn't provide any)

* ```ucm_to_tfrecord.ipynb``` converts the UCMerced land use dataset to a tensorflow record. It turns out it actually has one built in for this dataset so this is defunct.

* ```deep_feature_learning.ipynb``` contains code to build and train the model.

* ```predict_from_features.ipynb``` classifies the UCM dataset using a model trained in ```deep_feature_learning.ipynb```.

* ```predict_from_features_sentinel.ipynb``` does the same for the sentinel data.

* ```parameter_selection.ipynb``` runs all the saved models in the directory on the UCM data to select hyperparameters (We only changed the featureset volume and receptive field and otherwise stuck with the author's recommendations.

* ```download-data.sh``` and ```download-sentinel-data.sh```download the UCM and sentinel dataset into the correct folders respectively.

