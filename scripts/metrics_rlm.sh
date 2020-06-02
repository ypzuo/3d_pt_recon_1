#!/bin/bash

gpu=0
exp=trained_models/rlm
eval_set=valid
data_dir_imgs=data/shapenet/ShapeNetRendering
data_dir_pcl=data/shapenet/ShapeNet_pointclouds
declare -a categs=("vessel")

for cat in "${categs[@]}"; do
	echo python metrics_r.py --gpu $gpu --data_dir_imgs ${data_dir_imgs} --data_dir_pcl ${data_dir_pcl} --exp $exp --mode lm --category $cat --load_best --bottleneck 512 --bn_decoder --eval_set ${eval_set} --batch_size 24
	python metrics_r.py --gpu $gpu --data_dir_imgs ${data_dir_imgs} --data_dir_pcl ${data_dir_pcl} --exp $exp --mode lm --category $cat --load_best --bottleneck 512 --bn_decoder --eval_set ${eval_set} --batch_size 24
done

declare -a categs=("vessel")
for cat in "${categs[@]}"; do
	echo ${cat}
	cat ${exp}/metrics/${eval_set}/${cat}.csv
	echo
done
