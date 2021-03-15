# Repository for the Exposure team of the Guided Team Challange

__Table of contents:__  
1. Introduction
2. Project structure
3. xxxx

## 1. Introduction

This repository contains all code written for this challenge.

This project focuses on assessing change in the exposure of Caribbean informal settlements over time. This done firstly by segmenting satellite images to locate informal settlements, and then repeating this process at different times to determine change. This can identify growth or recession of informal settlements. 

Change detection algorithms were then developed, aiming to classify the effect of natural hazards on informal settlements, and hence determine a measure of vulnerability of these settlements. For example, following a disaster, change detection algorithms aim to determine the extent of damage suffered (e.g. destroyed, majorly damaged, undamaged). This was first approached with a ratio method, comparing the intensities of certain bands of pairs of satellite images to determine change. This simple method was built upon with a supervised deep learning approach, which was found to have limited success, likely due to the relatively low resolution of Sentinel-2 satellite imagery. To show the plausibility of such an approach, given high resolution data, a similar algorithm was applied to the labelled xBD dataset to classify damage sustained by buildings following a natural disaster.

This repository is split according to the structure of the write-up, with separate directories for settlement segmentation, change detection, and notebook that can be run to illustrate the different sections of the report.

## 2. Project Structure
```
├── LICENSE
├── README.md                   <- Main README.
|
├── notebooks                   <- Jupyter notebooks.
│
├── change_detection            <- code for change detection section.
│   ├──                         <- different subsections of change detection
│   │
│   ├──                         <- xxxx
│   │
│   ├──                         <- xxxx
|   |
│   ├──                         <- xxxx
│   │
│   └──                         <- xxxx
│
└── settlement_segmentation     <- settlement segmentation section
```

---
