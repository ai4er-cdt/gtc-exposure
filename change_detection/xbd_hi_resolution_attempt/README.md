# xBD Hi-Resolution Change Detection Attempt 

This code was adapted from the xView2 Baseline Model which was created to assess the difficulty of the xBD dataset and to serve as
a starting point for the xView 2 challenge. You can find their full version here: https://github.com/DIUx-xView/xView2_baseline.

You can find out more about the xView 2 challenge, which was hosted by the Defense Innovation Unit (DIU) with the goal of identifying buildings and rating the amount of damage they sustained from a natural disaster by using pre- and post satellite imagery, here: https://www.ibm.com/cloud/blog/the-xview2-ai-challenge.

The dataset presented with this challenge is the xBD dataset, which contains 850,736 annotated buildings and spans 45,362 square km of satellite imagery as well as over 9000 labelled pre- and post-disaster 1024x1024 images for training. You can download the dataset here: https://xview2.org/dataset.


### The Data Pipeline

The training data was reformatted in order to organise images and labels by disaster event in the following way.
```
xBD 
 ├── disaster_name_1
 │      ├── images 
 │      │      └── <image_id>.png
 │      │      └── ...
 │      ├── labels
 │      │      └── <image_id>.json
 │      │      └── ...
 ├── disaster_name_2
 │      ├── images 
 │      │      └── <image_id>.png
 │      │      └── ...
 │      ├── labels
 │      │      └── <image_id>.json
 │      │      └── ...
 └── disaster_name_n
```

You can do this with [`split_into_disasters.py`](./utils/split_into_disasters.py). This will take the location of your `train` directory and ask for an output directory to place each disaster with the subfolders `image/` and `label/`. 

Example call: 

`$ python split_into_disasters.py --input ~/Downloads/train/ --output ~/Downloads/xBD/`

### The Environment
Set up a conda environment with Python==3.7 using `conda activated $ENV_NAME python==3.7` and pip install the requirements.txt file.

You should also install tensorflow-gpu and pip if it is not already installed.

## The Training 

The training, was conducted on 1 node of the JASMIN GV100GL GPU hosts, and occured in two stages; first, localisation of building footpronts, followed by damage classification.
 
The below instructions are followed as in the [xView2 Baseline Model example](https://github.com/DIUx-xView/xView2_baseline).

#### Localization Training Pipeline

These are the pipeline steps are below for the instance segmentation training (these programs have been written and tested on Unix systems (Darwin, Fedora, Debian) only).

Note: As in the xView2 Baseline Model, a fork of motokimura's [SpaceNet Building Detection](https://github.com/motokimura/spacenet_building_detection) was used for our localization in order to automatically pull our polygons to feed into our classifier. 


First, we must create masks for the localization, and have the data in specific folders for the model to find and train itself. The steps we have built are described below:

1. Run `mask_polygons.py` to generate a mask file for the chipped images.
   * Sample call: `python mask_polygons.py --input /path/to/xBD --single-file --border 2`
   * Here border refers to shrinking polygons by X number of pixels. This is to help the model separate buildings when there are a lot of "overlapping" or closely placed polygons
   * Run `python mask_polygons.py --help` for the full description of the options. 
2. Run `data_finalize.sh` to setup the image and labels directory hierarchy that the `spacenet` model expects (will also run a python `compute_mean.py` to create a mean image our model uses during training. 
   * sample call: `data_finalize.sh -i /path/to/xBD/ -x /path/to/xView2/repo/root/dir/ -s .75`
   * -s is a crude train/val split, the decimal you give will be the amount of the total data to assign to training, the rest to validation
     * You can find this later in /path/to/xBD/spacenet_gt/dataSplit in text files, and easily change them after the script has been run. 
   * Run `data_finalize.sh` for the full description of the options. 
 
After these steps have been ran you will be ready for the instance segmentation training. 

The original images and labels are preserved in the `./xBD/org/$DISASTER/` directories, and just copies the images to the `spacenet_gt` directory.  

#### Training the SpaceNet Model

Now you will be ready to start training a model (based off our provided [weights](https://github.com/DIUx-xView/xview2-baseline/releases/tag/v1.0), or from a baseline).

Using the `spacenet` model we forked, you can control all of the options via command line calls.

In order for the model to find all of its required files, you will need to `$ cd /path/to/xView2/spacenet/src/models/` before running the training module. 

The main file is [`train_model.py`](./spacenet/src/models/train_model.py) and the options are below: 

Sample command: 
`$ python train_model.py /path/to/xBD/spacenet_gt/dataSet/ /path/to/xBD/spacenet_gt/images/ /path/to/xBD/spacenet_gt/labels/ -e 100`

**WARNING**: You must be in the `./spacenet/src/models/` directory to run the model due to relative paths for `spacenet` packages, as they have been edited for this instance and would be difficult to package. 

#### Damage Classification Training

The damage classification training processing and training code can be found under `/path/to/xView2/model/` 

You will need to run the `process_data.py` python script to extract the polygon images used for training, testing, and holdout from the original satellite images and the polygon labels produced by SpaceNet. This will generate a csv file with polygon UUID and damage type
as well as extracting the actual polygons from the original satellite images. If the `val_split_pct` is defined, then you will get two csv files, one for test and one for train. 

**Note** The process_data script only extracts polygons from post disaster images

After this, you can go ahead and train the damage classification.

Sample command: `$ python damage_classification.py --train_data /path/to/XBD/$process_data_output_dir/train 
--train_csv train.csv --test_data /path/to/XBD/$process_data_output_dir/test --test_csv test.csv --model_out path/to/xBD/baseline_trial --model_in /path/to/saved-model-01.hdf5`

### Inference 

Since the inference is done over two models we have created a script [inference.sh](./utils/inference.sh) to script together the inference code. 

If you would like to run the inference steps individually the shell script will provide you with those steps. 

To run the inference code you will need: 

1. The `xView2` repository (this one, where you are reading this README) cloned
2. An input pre image
3. An input post image that matches the pre in the same directory
4. Weights for the localization model
5. Weights for the classification model 

You can find my weights for both localisation in the inference folder.

As long as we can find the post image by replacing pre with post (`s/pre/post/g`) everything else should be run, this is used to dockerize the inference and run in parallel for each image individually based off the submission requirements of the challenge. 

Sample Call: `./utils/inference.sh -x /path/to/xView2/ -i /path/to/$DISASTER_$IMAGEID_pre_disaster.png -p  /path/to/$DISASTER_$IMAGEID_post_disaster.png -l /path/to/localization_weights.h5 -c /path/to/classification_weights.hdf5 -o /path/to/output/image.png -y`

### Results
<img width="1429" alt="Screenshot 2021-03-18 at 19 44 46" src="https://user-images.githubusercontent.com/53935327/111773798-8ad38a00-88a6-11eb-95fb-da0a1431b4e2.png">

