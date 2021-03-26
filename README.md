[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ai4er-cdt/gtc-exposure/HEAD)

![ESA](https://www.google.com/url?sa=i&url=https%3A%2F%2Fbrand.esa.int%2Fassets%2Flogo%2F&psig=AOvVaw3c8-NFFHfkmZ1FqfZxHyZl&ust=1616859435252000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCMD2j_Wkzu8CFQAAAAAdAAAAABAD)

# Repository for the Exposure team of the Guided Team Challenge

## 1. Overview

This repository contains all code written for this challenge.

This project focuses on assessing change in the exposure of Caribbean informal settlements over time. This is done firstly by segmenting satellite images to locate informal settlements, and then repeating this process at different times to determine change. Three different methods were used for image segmentation, a Random Forest model as well as two semi-supervised Deep Learning models. This can identify growth or recession of informal settlements. 

Change detection algorithms were then developed, aiming to classify the effect of natural hazards on informal settlements, and hence determine a measure of vulnerability of these settlements. For example, following a disaster, change detection algorithms aim to determine the extent of damage suffered (e.g. destroyed, majorly damaged, undamaged). This was first approached with a ratio method, comparing the intensities of certain bands of pairs of satellite images to determine change. This simple method was built upon with a supervised deep learning approach, which was found to have limited success, likely due to the relatively low resolution of Sentinel-2 satellite imagery. To show the plausibility of such an approach, given high resolution data, a similar algorithm was applied to the labelled xBD dataset to classify damage sustained by buildings following a natural disaster.

This repository is split according to the structure of the write-up, with separate directories for settlement segmentation, change detection, and exposure quantification. Each contain notebooks that can be run to illustrate the different sections of the report.

## 2. Project Structure
```
├── LICENSE
├── README.md                   <- Main README.
├── settlement_segmentation     <- Settlement segmentation section.
│   │
│   ├── deepcluster             <- DeepCluster model as well as training and testing notebooks
│   │
│   ├── liunsupervised          
|   |
|   └── randomforest            
|
├── change_detection            <- Change detection section.
│   ├── archive                 <- Archive of old code for this section
│   │
│   ├── deep_change_detection   <- Code for the deep learning approach to change detection on Sentinel-2 data.
│   │
│   ├── ratio_method            <- Code for the image ratio methods including thresholding and U-Net classification
|   |
|   └── xbd_hi_res_attempt      <- Code for semantic change detection applied to high resolution data
|
└── exposure_quantification

```

---
