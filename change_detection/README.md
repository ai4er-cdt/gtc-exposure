# Change detection

Following on from settlement segmentation, it would be useful to have some measure of change following a natural disaster to ascertain the vulnerability of informal settlements. This can be determined by analysing before and after images of informal settlements affected by natural hazards and assessing the damage.

This was approached firstly via a simple, mathematical method, comparing a pair of images, and determining whether the intensity of colour has changed significantly between them. The code for this method can be found in ratio_attempt.

While this method is shown to be effective in detecting change, it is not designed for assessing any kind of semantic change (e.g. degree of damage sustained by a building following a natural disaster). We therefore turned to a supervised deep learning method, trained on labelled images from the OSCD dataset. This approach was motivated by reading the literature and determining that supervised deep learning showed the most promise. A recent unsupervised deep learning approach (Leenstra et al., 2021) was found to only marginally improve on the supervised method explored here. 

The code for our supervised approach was adapted from the state-of-the-art approach on Sentinel-2 satellite data, found in Daudt et al. (2018).  The code for this method can be found in deep_change_detection. We trained a model using both UNet and Siamese architectures, and found UNet to be most effective, achieving 0.45 F1-score on the OSCD test set. The weights of the trained model are in unet_final_weights.pth.tar. To see how effective the model is, example .png files over various cities have been uploaded and can be seen in the results directory. In these images, true positives are white, false negative are green, false positives are magenta, and true negatives are black.

We have also produced a pipeline, to be run on the Descartes Labs platform to run change detection on any location worldwide, using both the ratio and supervised deep learning method.

Finally, to show what is possible, given a larger, higher resolution dataset, we devised a similar approach on the xBD dataset (https://xview2.org/dataset).


OSCD dataset downloadable at https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection

Rodrigo Caye Daudt, Bertrand Le Saux, Alexandre Boulch. (2018, October). Fully convolutional siamese networks for change detection. In 2018 25th IEEE International Conference on Image Processing (ICIP) (pp. 4063-4067). IEEE.](https://ieeexplore.ieee.org/abstract/document/8451652)
[arXiv](https://arxiv.org/abs/1810.08462)

Leenstra, M., Marcos, D., Bovolo, F. and Tuia, D., 2021. Self-supervised pre-training enhances change detection in Sentinel-2 imagery. arXiv preprint arXiv:2101.08122.
