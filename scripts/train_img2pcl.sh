python train_img2pcl.py \
	--mode lm \
	--data_dir_imgs data/shapenet/ShapeNetRendering \
	--data_dir_pcl data/shapenet/ShapeNet_pointclouds \
	--exp expts/img2pcl_car \
	--gpu 0 \
	--category car \
    --bottleneck 512 \
	--loss chamfer \
	--batch_size 128 \
	--lr 5e-5 \
	--bn_decoder \
	--max_epoch 5 \
	--print_n 100
	# --sanity_check
