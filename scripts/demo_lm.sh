#!/bin/bash

gpu=0
exp=trained_models/lm
data_dir_imgs=data/shapenet/ShapeNetRendering
data_dir_pcl=data/shapenet/ShapeNet_pointclouds
eval_set=valid
cat=car

echo python metrics.py --gpu $gpu --data_dir_imgs ${data_dir_imgs} --data_dir_pcl ${data_dir_pcl} --exp $exp --mode lm --category $cat --load_best --bottleneck 512 --bn_decoder --eval_set ${eval_set} --batch_size 24 --visualize
python metrics.py --gpu $gpu --data_dir_imgs ${data_dir_imgs} --data_dir_pcl ${data_dir_pcl} --exp $exp --mode lm --category $cat --load_best --bottleneck 512 --bn_decoder --eval_set ${eval_set} --batch_size 24 --visualize

