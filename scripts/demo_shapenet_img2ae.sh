#!/bin/bash

gpu=0
exp=expts/img2pcl_chair
ae_exp=expts/now/pure_ae_1024
dataset=shapenet
data_dir_imgs=data/shapenet/ShapeNetRendering
data_dir_pcl=data/shapenet/ShapeNet_pointclouds
eval_set=valid
cat=chair

python metrics_img2ae.py \
	--gpu $gpu \
	--dataset $dataset \
	--data_dir_imgs ${data_dir_imgs} \
	--data_dir_pcl ${data_dir_pcl} \
	--exp $exp \
	--mode lm \
	--category $cat \
	--load_best \
	--bottleneck 512 \
	--ae_logs $ae_exp \
	--bn_decoder \
	--eval_set ${eval_set} \
	--batch_size 24 \

