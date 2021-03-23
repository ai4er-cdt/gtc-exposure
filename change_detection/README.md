# Change Detection

Once identified through segmentation, we need to assess the temporal variation in exposure and vulnerability of informal settlements. This requires detecting material change between subsequent satellite images to deduce exposure from building location and infer vulnerability from changes in building integrity, notably following disasters.

## 1 - Ratio Method
### Thresholding ([notebook](https://github.com/ai4er-cdt/gtc-exposure/blob/main/change_detection/ratio_method/thresholding.ipynb))

Thresholding the ratio of pixel values between images from two subsequent dates is a simple method for detecting change in that pixel.Due to ground truth damage assessment data availability from [Copernicus Emergency Management Service](https://emergency.copernicus.eu/mapping/map-of-activations-rapid#zoom=2&lat=31.15459&lon=1.92545&layers=BT00), we calibrate this technique using Sentinel-2 pre- and post-disaster imagery. After parameter optimisation, thresholding achieves average precision and F1-scores of 0.82 and 0.54 on our test set, showing effectiveness in rapidly identifying change from free and frequently updated Sentinel-2 imagery, providing a useful initial assessment in re-evaluating a settlement's exposure or vulnerability.

The principle accuracy limitations are the 10m resolution of Sentinel-2 imagery, precluding small-scale change detection, and ratio value sensitivity to image lighting, triggering false detections if the threshold is unadjusted.

### U-Net Classification ([notebook](https://github.com/ai4er-cdt/gtc-exposure/blob/main/change_detection/ratio_method/unet_classifier.ipynb))

To reduce lighting sensitivity, we trained a U-Net classification model to recognise change within image ratios. It shows good performance in training locations. Test set precision and F1 scores of 0.91 and 0.46 indicate poorer generalisation than thresholding. However, avoiding threshold adjustments is advantageous.

## 2 - Deep Learning Approach ([notebook](https://github.com/ai4er-cdt/gtc-exposure/blob/main/change_detection/deep_change_detection/cd_dl_notebook.ipynb))

While the ratio method is shown to be effective in detecting change, it is not designed for assessing any kind of semantic change (e.g. degree of damage sustained by a building following a natural disaster). We therefore turned to a supervised deep learning method, trained on labelled images from the OSCD dataset. This approach was motivated by reading the literature and determining that supervised deep learning showed the most promise. A recent unsupervised deep learning approach (Leenstra et al., 2021) was found to only marginally improve on the supervised method explored here. 

The code for our supervised approach was adapted from the state-of-the-art approach on Sentinel-2 satellite data, found in Daudt et al. (2018).  The code for this method can be found in deep_change_detection. We trained a model using both UNet and Siamese architectures, and found UNet to be most effective, achieving 0.45 F1-score on the OSCD test set. The weights of the trained model are in unet_final_weights.pth.tar. To see how effective the model is, example .png files over various cities have been uploaded and can be seen in the results directory. In these images, true positives are white, false negative are green, false positives are magenta, and true negatives are black.

We have also produced a pipeline, to be run on the Descartes Labs platform to run change detection on any location worldwide, using both the ratio and supervised deep learning method.

## 3 - High Resolution Example

Finally, to show what is possible, given a larger, higher resolution dataset, we devised a similar approach on the xBD dataset (https://xview2.org/dataset).

The method was split into two stages, first, building localisation was performed to identify individual buildings, followed by damage classification to assign a level of damage to each building pre- and post-disaster. The model performed well despite encountered limitations, achieving an accuracy of 0.97 and F1 score of 0.40 on the localisation and classification runs respectively.

xBD dataset downloadable at 

## Directory Structure
```
├── README.md                    <- This README.
├── archive                      <- Old code for this section
├── deep_change_detection        <- Code for the deep learning approach to change detection on Sentinel-2 data.
│   ├── example_images           <- Example images for testing
│   ├── weights                  <- Weights files from trained models
│   └──  cd_dl_notebook.ipynb    <- Deep learning approach notebook
│   
├── ratio_method                 <- Code for the ratio method with thresholding and U-Net classifier notebooks
│   ├── coastlines               <- Geojsons for ocean masking used in thresholding
│   ├── geojsons                 <- Geojsons containing building footprints used in both notebooks
│   ├── gradings                 <- Copernicus EMS bulding damage assessments
│   ├── models                   <- Trained model weights for unet_classifier
│   ├── records                  <- Satellite imagery patches corresponding to building footprints
│   ├── thresholding.ipynb       <- Thresholding Method notebook
│   └── unet_classifier.ipynb    <- U-Net Classification notebook
|
├── xbd_hi_res_attempt           <- Code for semantic change detection applied to high resolution data
│   ├── inference
│   ├── model
│   ├── overlay_output_to_image
│   ├── spacenet
│   ├── utils
│   ├── weights
└── └── requirements.txt

```

Rodrigo Caye Daudt, Bertrand Le Saux, Alexandre Boulch. (2018, October). Fully convolutional siamese networks for change detection. In 2018 25th IEEE International Conference on Image Processing (ICIP) (pp. 4063-4067). IEEE.](https://ieeexplore.ieee.org/abstract/document/8451652)
[arXiv](https://arxiv.org/abs/1810.08462)

Leenstra, M., Marcos, D., Bovolo, F. and Tuia, D., 2021. Self-supervised pre-training enhances change detection in Sentinel-2 imagery. arXiv preprint arXiv:2101.08122.
