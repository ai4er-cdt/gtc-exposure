This directory corresponds to the DeepCluster model used for unsupervised settlement classification in the report. The model is writen using PyTorch.

* The different base CNN models can be found within the ```models``` directory.

* The ```train.ipynb``` notebook runs the shell command to train the model and data paths as well as training parameters can be specified within the notebook.

* The ```test.ipynb``` notebook evaulates the trained model on the test dataset and can visualise the cluster assingments as well as calculate accuracy and f1 scores.


The paper from which this architeture is based can be founf here: [Deep Clustering for Unsupervised Learning of Visual Features](https://arxiv.org/abs/1807.05520).
