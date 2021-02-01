# Change detection

Fully convolutional network architectures for change detection using sentinel-2 images.

Notebook is one of the files - please look at this to see what is going on.

We trained a model using UNet.py, which achieves 91% accuracy on the OSCD test set. The weights of the trained model are in net_final.pth.tar . To see how effective the model is, an example .png over a city has been uploaded and can be seen at FC-EF-chongqing.png. In this image, true positives are white, false negative are green, false positives are magenta, and true negatives are black.


Using OSCD dataset downloadable at https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection

Drawn from this paper. Some adaptions to try it on a CPU - unsurprisingly this was infeasible even with a tiny training dataset and network. 

[Rodrigo Caye Daudt, Bertrand Le Saux, Alexandre Boulch. (2018, October). Fully convolutional siamese networks for change detection. In 2018 25th IEEE International Conference on Image Processing (ICIP) (pp. 4063-4067). IEEE.](https://ieeexplore.ieee.org/abstract/document/8451652)

[arXiv](https://arxiv.org/abs/1810.08462)
