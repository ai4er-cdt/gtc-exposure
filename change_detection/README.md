# Change detection

Following on from settlement segmentation, it would be useful to have some measure of change following a natural disaster to ascertain the vulnerability of informal settlements. This can be determined by analysing before and after images of areas affected by natural hazards and then assessing the damage.

This was approached firstly via a simple, mathematical method, comparing a pair of images, and determining whether the intensity of colour has changed significantly between them. The code for this method can be found in ratio_attempt.

While this method is shown to be effective in detecting change, it is not designed for assessing any kind of semantic change (e.g. degree of damage sustained by a building following a natural disaster). We therefore turned to a supervised deep learning method, trained on labelled images from the OSCD dataset, to ascertain firstly whether artificial change (non-natural, e.g. forest growth/cutting, cloud appearance etc.) had occurred. The code for this was adapted from the state-of-the-art approach on Sentinel-2 satellite data, found in Daudt et al. (2018).  The code for this method can be found in deep_change_detection. We trained a model using both UNet and Siamese architectures, and found UNet to be most effective, achieving 0.45 F1-score on the OSCD test set. The weights of the trained model are in net_final.pth.tar. To see how effective the model is, example .png files over various cities have been uploaded and can be seen in the results directory. In this image, true positives are white, false negative are green, false positives are magenta, and true negatives are black.

We have also produced a pipeline, to be run on the Descartes Labs platform to run change detection on any location worldwide, using both the ratio and deep learning method.

The limiting feature in this work was determined to be the resolution (10m) of the Sentinel satellite data. 

To show what is possible, given higher resolution data, we devised a similar approach on the labelled xBD dataset.




OSCD dataset downloadable at https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection

Model for the UNet and Siamese approaches are taken from this paper.

[Rodrigo Caye Daudt, Bertrand Le Saux, Alexandre Boulch. (2018, October). Fully convolutional siamese networks for change detection. In 2018 25th IEEE International Conference on Image Processing (ICIP) (pp. 4063-4067). IEEE.](https://ieeexplore.ieee.org/abstract/document/8451652)
[arXiv](https://arxiv.org/abs/1810.08462)
