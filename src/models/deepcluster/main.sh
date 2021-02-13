# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DIR="/home/ira/Documents/gtc-exposure/notebooks/is500/deepcluster/datasets/"
ARCH="alexnet"
LR=0.05
WD=-5
K=10000
WORKERS=12
EXP="/home/ira/Documents/gtc-exposure/notebooks/is500/deepcluster/output/"
PYTHON="/usr/bin/python3"

mkdir -p ${EXP}

# CUDA_VISIBLE_DEVICES="" 
${PYTHON} main.py ${DIR} --exp ${EXP} --arch ${ARCH} \
  --lr ${LR} --wd ${WD} --k ${K} --sobel --verbose --workers ${WORKERS}
