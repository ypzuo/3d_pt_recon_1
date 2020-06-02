from importer import *
import utils.pointnet2_utils.tf_util
from utils.pointnet2_utils.pointnet_util import pointnet_sa_module, pointnet_fp_module, pointnet_sa_module_msg
from utils.pointnet2_utils.pointSIFT_util import pointSIFT_module
parser = argparse.ArgumentParser()

parser.add_argument('--data_dir_pcl', type=str, required=True, 
	help='Path to shapenet pointclouds')
parser.add_argument('--exp', type=str, required=True, 
	help='Name of Experiment')
parser.add_argument('--gpu', type=str, required=True, 
	help='GPU to use')
parser.add_argument('--category', type=str, required=True, 
	help='Category to train on : \
		["all", "airplane", "bench", "cabinet", "car", "chair", "lamp", \
		"monitor", "rifle", "sofa", "speaker", "table", "telephone", "vessel"]')
parser.add_argument('--in_pcl_size', type=int, required=True, 
	help='Size of input point cloud')
parser.add_argument('--bottleneck', type=int, required=True, default=512, 
	help='latent space size')
parser.add_argument('--batch_size', type=int, default=32, 
	help='Batch Size during training')
parser.add_argument('--lr', type=float, default=0.0005, 
	help='Learning Rate')
parser.add_argument('--max_epoch', type=int, default=500, 
	help='max num of epoch')
parser.add_argument('--bn_decoder', action='store_true', 
	help='Supply this parameter if you want batch norm in the decoder, otherwise ignore')
parser.add_argument('--print_n', type=int, default=100, 
	help='print output to terminal every n iterations')
parser.add_argument('--localsift', action='store_true', 
	help='Supply this parameter if you want to use localsift point feature, otherwise ignore')

FLAGS = parser.parse_args()

print '-='*50
print FLAGS
print '-='*50

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
IN_PCL_SIZE = FLAGS.in_pcl_size
OUT_PCL_SIZE = FLAGS.in_pcl_size*2
BATCH_SIZE = FLAGS.batch_size  		# Batch size for training
NUM_POINTS = 2048					# Number of predicted points
GT_PCL_SIZE = 16384					# Number of points in GT point cloud
UPSAMPLING_FACTOR = 2


def fetch_batch(models, batch_num, batch_size):
	'''
	Input:
		models: list of paths to shapenet models
		batch_num: batch_num during epoch
		batch_size:	batch size for training or validation
	Returns:
		batch_gt: (B,2048,3)
	Description:
		Batch Loader
	'''

	batch_gt = []
	for ind in range(batch_num*batch_size, batch_num*batch_size+batch_size):
		model_path = models[ind]
		pcl_path = join(FLAGS.data_dir_pcl, model_path, 'pointcloud_2048.npy') # Path to 2K ground truth point cloud
		pcl_gt = np.load(pcl_path)
		batch_gt.append(pcl_gt)
	batch_gt = np.array(batch_gt)
	return batch_gt

def fetch_batch_dense(models, batch_num, batch_size):
	batch_ip = []
	batch_gt = []
	for ind in range(batch_num*batch_size, batch_num*batch_size+batch_size):
		model_path = models[ind]
		pcl_gt = np.load(join(FLAGS.data_dir_pcl, model_path, 'pointcloud_2048.npy'))
		pcl_ip = np.load(join(FLAGS.data_dir_pcl, model_path,'pointcloud_1024.npy'))
		batch_gt.append(pcl_gt)
		batch_ip.append(pcl_ip)
	batch_gt = np.array(batch_gt)
	batch_ip = np.array(batch_ip)
	return batch_ip, batch_gt


def get_epoch_loss(val_models):

	'''
	Input:
		val_models:	list of absolute path to models in validation set
	Returns:
		val_chamfer: chamfer distance calculated on scaled prediction and gt
		val_forward: forward distance calculated on scaled prediction and gt
		val_backward: backward distance calculated on scaled prediction and gt
	Description:
		Calculate val epoch metrics
	'''
	
	tflearn.is_training(False, session=sess)

	batches = len(val_models)/BATCH_SIZE
	val_stats = {}
	val_stats = reset_stats(ph_summary, val_stats)

	for b in xrange(batches):
		batch_ip, batch_gt = fetch_batch_dense(val_models, b, BATCH_SIZE)
		runlist = [loss, chamfer_distance_scaled, dists_forward_scaled, dists_backward_scaled]
		L,C,F,B = sess.run(runlist, feed_dict={pcl_in:batch_ip, pcl_gt:batch_gt})
		_summary_losses = [L, C, F, B]

		val_stats = update_stats(ph_summary, _summary_losses, val_stats, batches)

	summ = sess.run(merged_summ, feed_dict=val_stats)
	return val_stats[ph_dists_chamfer], val_stats[ph_dists_forward], val_stats[ph_dists_backward], summ


if __name__ == '__main__':

	# Create a folder for experiments and copy the training file
	create_folder(FLAGS.exp)
	train_filename = basename(__file__)
	os.system('cp %s %s'%(train_filename, FLAGS.exp))
	with open(join(FLAGS.exp, 'settings.txt'), 'w') as f:
		f.write(str(FLAGS)+'\n')

	# Create Placeholders
	#pcl_gt = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINTS, 3))
	pcl_in = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IN_PCL_SIZE, 3), name='pcl_in')
	pcl_gt = tf.placeholder(tf.float32, shape=(BATCH_SIZE, OUT_PCL_SIZE, 3), name='pcl_gt')

	# Generate Prediction
	bneck_size = FLAGS.bottleneck
	with tf.variable_scope('pointnet_ae') as scope:
		z = encoder_with_convs_and_symmetry(in_signal=pcl_in, n_filters=[32,64,64], 
			filter_sizes=[1],
			strides=[1],
			b_norm=True,
			verbose=True,
			scope=scope
			)
		print 'z: ', z.shape
		print 'pcl_in: ', pcl_in.shape
		point_feat = tf.tile(tf.expand_dims(pcl_in, 2), [1,1,UPSAMPLING_FACTOR,1]) # (bs,NUM_POINTS,3) --> (bs,NUM_POINTS,1,3) --> (bs,NUM_POINTS,4,3)
		point_feat = tf.reshape(point_feat, [BATCH_SIZE, OUT_PCL_SIZE, 3]) # (bs,NUM_POINTS,4,3) --> (bs,NUM_UPSAMPLE_POINTS,3)
		print 'point_feat: ', point_feat.shape
		global_feat = tf.expand_dims(z, axis=1) # (bs,bneck) --> (bs,1,bneck)
		print 'global_feat: ', global_feat.shape
		global_feat = tf.tile(global_feat, [1, OUT_PCL_SIZE, 1]) # (bs,1,bneck) --> (bs,NUM_UPSAMPLE_POINTS,bneck)
		print 'global_feat: ', global_feat.shape
		concat_feat = tf.concat([point_feat, global_feat], axis=2)
		print 'concat_feat: ', concat_feat.shape
		if FLAGS.localsift:
			#pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True)
			_, local_pt, _ = pointnet_sa_module(pcl_in, None, npoint=IN_PCL_SIZE, radius=0.1, nsample=8, mlp=[32,32,64], mlp2=None, group_all=False, is_training=False, bn_decay=None, scope='localsift_feat', bn=False)
			_, local_feat, _ = pointSIFT_module(pcl_in, local_pt, 0.25, out_channel=8, is_training=False, bn_decay=None, scope='local_feat', bn=False)
			#pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', sort_k=None, tnet_spec=None, knn=False, use_xyz=True):
			#pointSIFT_module(xyz, points, radius, out_channel, is_training, bn_decay, scope='point_sift', bn=True, use_xyz=True):
			print 'local_feat: ', local_feat.shape
			local_feat = tf.tile(tf.expand_dims(local_feat, 2), [1,1,UPSAMPLING_FACTOR,1]) # (bs,NUM_POINTS,64) --> (bs,NUM_POINTS,1,64) --> (bs,NUM_POINTS,4,64)
			local_feat = tf.reshape(local_feat, [BATCH_SIZE, OUT_PCL_SIZE, -1]) # (bs,NUM_POINTS,4,64) --> (bs,NUM_UPSAMPLE_POINTS,64)
			concat_feat = tf.concat([concat_feat, local_feat], axis=2)
			print 'concat_feat: ', concat_feat.shape
			z = concat_feat
		#out = decoder_with_fc_only(z, layer_sizes=[64,128,128,np.prod([NUM_POINTS, 3])],
			#b_norm=FLAGS.bn_decoder,
			#b_norm_finish=False,
			#verbose=True,
			#scope=scope
			#)
		out = decoder_with_convs_only(z, n_filters=[64,128,128,3], 
			filter_sizes=[1], 
			strides=[1],
			b_norm=FLAGS.bn_decoder, 
			b_norm_finish=False, 
			verbose=True, 
			scope=scope)
		print 'final out: ', out.shape
		out = tf.reshape(out, (BATCH_SIZE, NUM_POINTS, 3))

	# Scale output and gt for val losses
	pcl_gt_scaled, out_scaled = scale(pcl_gt, out)
	
	# Calculate Chamfer Metrics
	dists_forward, dists_backward, chamfer_distance = [tf.reduce_mean(metric) for metric in get_chamfer_metrics(pcl_gt, out)]

	# Calculate Chamfer Metrics on scaled prediction and GT
	dists_forward_scaled, dists_backward_scaled, chamfer_distance_scaled = [tf.reduce_mean(metric) for metric in get_chamfer_metrics(pcl_gt_scaled, out_scaled)]

	# Define Loss to optimize on
	loss = (dists_forward + dists_backward/2.0)*10000

	# Get Training Models
	train_models, val_models, _, _ = get_shapenet_models(FLAGS)
	batches = len(train_models) / BATCH_SIZE

	# Training Setings
	lr = FLAGS.lr
	optim = tf.train.AdamOptimizer(FLAGS.lr, beta1=0.9).minimize(loss)

	start_epoch = 0
	max_epoch = FLAGS.max_epoch

	# Define Log Directories
	snapshot_folder = join(FLAGS.exp, 'snapshots')
	best_folder = join(FLAGS.exp, 'best')
	logs_folder = join(FLAGS.exp, 'logs')	

	# Define Savers
	saver = tf.train.Saver(max_to_keep=2)

	# Define Summary Placeholders
	ph_loss = tf.placeholder(tf.float32, name='loss')
	ph_dists_chamfer = tf.placeholder(tf.float32, name='dists_chamfer')
	ph_dists_forward = tf.placeholder(tf.float32, name='dists_forward')
	ph_dists_backward = tf.placeholder(tf.float32, name='dists_backward')

	ph_summary = [ph_loss, ph_dists_chamfer, ph_dists_forward, ph_dists_backward]
	merged_summ = get_summary(ph_summary)

	# Create log directories
	create_folders([snapshot_folder, logs_folder, join(snapshot_folder, 'best'), best_folder])

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	with tf.Session(config=config) as sess:

		train_writer = tf.summary.FileWriter(logs_folder+'/train', sess.graph_def)
		val_writer = tf.summary.FileWriter(logs_folder+'/val', sess.graph_def)

		sess.run(tf.global_variables_initializer())

		# Load Previous checkpoint
		start_epoch = load_previous_checkpoint(snapshot_folder, saver, sess)

		ind = 0
		best_val_loss = 10000000
		since = time.time()

		print '*'*30,'\n','Training Started !!!\n', '*'*30

		PRINT_N = FLAGS.print_n

		for i in xrange(start_epoch, max_epoch): 
			random.shuffle(train_models)
			stats = {}
			stats = reset_stats(ph_summary, stats)
			iter_start = time.time()

			tflearn.is_training(True, session=sess)

			for b in xrange(batches):
				global_step = i*batches + b + 1
				batch_ip, batch_gt = fetch_batch_dense(train_models, b, BATCH_SIZE)

				runlist = [loss, chamfer_distance, dists_forward, dists_backward, optim]
				L, C, F, B, _ = sess.run(runlist, feed_dict={pcl_in:batch_ip, pcl_gt:batch_gt})
				_summary_losses = [L, C, F, B]

				stats = update_stats(ph_summary, _summary_losses, stats, PRINT_N)

				if global_step % PRINT_N == 0:
					summ = sess.run(merged_summ, feed_dict=stats)
					train_writer.add_summary(summ, global_step)
					till_now = time.time() - iter_start
					print 'Loss = {} Iter = {}  Minibatch = {} Time:{:.0f}m {:.0f}s'.format(
						stats[ph_loss], global_step, b, till_now//60, till_now%60
					)
					stats = reset_stats(ph_summary, stats)
					iter_start = time.time()

			print 'Saving Model ....................'
			saver.save(sess, join(snapshot_folder, 'model'), global_step=i)
			print '..................... Model Saved'

			val_epoch_chamfer, val_epoch_forward, val_epoch_backward, val_summ = get_epoch_loss(val_models)
			val_writer.add_summary(val_summ, global_step)

			time_elapsed = time.time() - since

			print '-'*65 + ' EPOCH ' + str(i) + ' ' + '-'*65
			print 'Val Chamfer: {:.8f}  Forward: {:.8f}  Backward: {:.8f}  Time:{:.0f}m {:.0f}s'.format(
				val_epoch_chamfer, val_epoch_forward, val_epoch_backward, time_elapsed//60, time_elapsed%60
			)
			print '-'*140
			print

			if (val_epoch_chamfer < best_val_loss):
				print 'Saving Best at Epoch %d ...............'%(i)
				saver.save(sess, join(snapshot_folder, 'best', 'best'))
				os.system('cp %s %s'%(join(snapshot_folder, 'best/*'), best_folder))
				best_val_loss = val_epoch_chamfer
				print '.............................Saved Best'
