Generates training data from the cloud-free mosiac Sentinel-2 satellite imagery. 
The download and preprocessing steps can be completed by running though the ```cloud_free.ipynb``` notebook.

In order to change the polygon areas of interest to cut from full dataset from the Carribean, this can be done by changing the ```polygon```, ```informal_settlement``` and ```test_area``` functions in ```cloud_free.py```.

```pytorch_preprocess.py``` creates data images suitable to be passed into the DeepCluster model found in [```../../deepcluster```](https://github.com/ai4er-cdt/gtc-exposure/tree/main/settlement_segmentation/deepcluster).

Data available [here](http://data.europa.eu/89h/0bd1dfab-e311-4046-8911-c54a8750df79).
