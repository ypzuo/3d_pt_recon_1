from importer import *

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir_imgs', type=str, required=True, 
	help='Path to shapenet rendered images')
parser.add_argument('--data_dir_pcl', type=str, required=True, 
	help='Path to shapenet pointclouds')
parser.add_argument('--exp', type=str, required=True, 
	help='Name of Experiment')
parser.add_argument('--exp_psgn', type=str, required=True, 
	help='Name of psgn Experiment')
parser.add_argument('--exp_ae', type=str, required=True, 
	help='Name of pointnet_ae Experiment')
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
#OUT_PCL_SIZE = FLAGS.in_pcl_size*2
BATCH_SIZE = FLAGS.batch_size  		# Batch size for training
NUM_POINTS = 2048					# Number of predicted points
GT_PCL_SIZE = 16384					# Number of points in GT point cloud
UPSAMPLING_FACTOR = 2
HEIGHT = 128 						# Height of input RGB image
WIDTH = 128 						# Width of input RGB image

def fetch_batch(models, indices, batch_num, batch_size):
	'''
	Input:
		models: list of paths to shapenet models
		indices: list of ind pairs, where 
			ind[0] : model index (range--> [0, len(models)-1])
			ind[1] : view index (range--> [0, NUM_VIEWS-1])
		batch_num: batch_num during epoch
		batch_size: batch size for training or validation
	Returns:
		batch_ip: input RGB image of shape (B, HEIGHT, WIDTH, 3)
		batch_gt: gt point cloud of shape (B, NUM_POINTS, 3)
	Description:
		Batch Loader
	'''

	batch_ip = []
	batch_gt = []

	for ind in indices[batch_num*batch_size:batch_num*batch_size+batch_size]:
		model_path = models[ind[0]]
		img_path = join(FLAGS.data_dir_imgs, model_path, 'rendering', PNG_FILES[ind[1]])
		pcl_path = join(FLAGS.data_dir_pcl, model_path, 'pointcloud_2048.npy')

		pcl_gt = np.load(pcl_path)

		ip_image = cv2.imread(img_path)[4:-5, 4:-5, :3]
		ip_image = cv2.cvtColor(ip_image, cv2.COLOR_BGR2RGB)

		batch_gt.append(pcl_gt)
		batch_ip.append(ip_image)

	return np.array(batch_ip), np.array(batch_gt)

def get_epoch_loss(val_models, val_pair_indices):

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

	batches = len(val_pair_indices)/BATCH_SIZE
	val_stats = {}
	val_stats = reset_stats(ph_summary, val_stats)

	for b in xrange(batches):
		batch_ip, batch_gt = fetch_batch(val_models, val_pair_indices, b, BATCH_SIZE)
		runlist = [loss, chamfer_distance_scaled, dists_forward_scaled, dists_backward_scaled]
		L,C,F,B = sess.run(runlist, feed_dict={ pcl_gt:batch_gt, img_inp: batch_ip})
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
	img_inp = tf.placeholder(tf.float32, shape=(BATCH_SIZE, HEIGHT, WIDTH, 3), name='img_inp')
	pcl_gt = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINTS, 3), name='pcl_gt')
	# Generate Prediction
	bneck_size = FLAGS.bottleneck

	with tf.variable_scope('psgn') as scope:
		z_latent_img = image_encoder(img_inp, FLAGS)

		out_img = decoder_with_fc_only(z_latent_img, layer_sizes=[256,256,np.prod([NUM_POINTS, 3])],
			b_norm=FLAGS.bn_decoder,
			b_norm_finish=False,
			verbose=True,
			scope=scope
			)

		# Point cloud reconstructed from input RGB image using latent matching network and fixed decoder from AE 
		reconstr_img = tf.reshape(out_img, (BATCH_SIZE, NUM_POINTS, 3))

	with tf.variable_scope('pointnet_ae') as scope:
		z = encoder_with_convs_and_symmetry(in_signal=reconstr_img, n_filters=[64,128,128,256,bneck_size], 
			filter_sizes=[1],
			strides=[1],
			b_norm=True,
			verbose=True,
			scope=scope
			)
		out = decoder_with_fc_only(z, layer_sizes=[256,256,np.prod([NUM_POINTS, 3])],
			b_norm=FLAGS.bn_decoder,
			b_norm_finish=False,
			verbose=True,
			scope=scope
			)
		out = tf.reshape(out, (BATCH_SIZE, NUM_POINTS, 3))

	#Get the AE vars
	pointnet_ae_vars = [var for var in tf.global_variables() if 'pointnet_ae' in var.name]
	psgn_vars = [var for var in tf.global_variables() if 'psgn' in var.name]
	train_vars = pointnet_ae_vars + psgn_vars

	# Scale output and gt for val losses
	pcl_gt_scaled, out_scaled = scale(pcl_gt, out)
	
	# Calculate Chamfer Metrics
	dists_forward, dists_backward, chamfer_distance = [tf.reduce_mean(metric) for metric in get_chamfer_metrics(pcl_gt, out)]

	# Calculate Chamfer Metrics on scaled prediction and GT
	dists_forward_scaled, dists_backward_scaled, chamfer_distance_scaled = [tf.reduce_mean(metric) for metric in get_chamfer_metrics(pcl_gt_scaled, out_scaled)]

	# Define Loss to optimize on
	#loss = (dists_forward + dists_backward/2.0)*10000
	loss = (dists_forward + dists_backward/2.0)*10000

	# Get Training Models
	train_models, val_models, train_pair_indices, val_pair_indices = get_shapenet_models(FLAGS)
	batches = len(train_pair_indices) / BATCH_SIZE

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
	saver = tf.train.Saver(max_to_keep=2, var_list=train_vars)
	saver_psgn = tf.train.Saver(psgn_vars)
	saver_ae = tf.train.Saver(pointnet_ae_vars)

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
		#start_epoch = load_previous_checkpoint(snapshot_folder, saver, sess)
		ckpt = tf.train.get_checkpoint_state(snapshot_folder)
		if ckpt is not None:
			print ('loading '+ckpt.model_checkpoint_path + '  ....')
			saver.restore(sess, ckpt.model_checkpoint_path)
			start_epoch = int(re.match('.*-(\d*)$', ckpt.model_checkpoint_path).group(1))

		# Load pretrained models
		if start_epoch == 0:
			load_previous_finetune(join(FLAGS.exp_ae, 'best', 'best'), saver_ae, sess)
			load_previous_finetune(join(FLAGS.exp_psgn, 'best', 'best'), saver_psgn, sess)



		ind = 0
		best_val_loss = 10000000
		since = time.time()

		print '*'*30,'\n','Training Started !!!\n', '*'*30

		PRINT_N = FLAGS.print_n

		for i in xrange(start_epoch, max_epoch): 
			random.shuffle(train_pair_indices)
			stats = {}
			stats = reset_stats(ph_summary, stats)
			iter_start = time.time()

			tflearn.is_training(True, session=sess)

			for b in xrange(batches):
				global_step = i*batches + b + 1
				batch_ip, batch_gt = fetch_batch(train_models, train_pair_indices, b, BATCH_SIZE)

				runlist = [loss, chamfer_distance, dists_forward, dists_backward, optim]
				L, C, F, B, _ = sess.run(runlist, feed_dict={pcl_gt:batch_gt, img_inp: batch_ip})
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

			val_epoch_chamfer, val_epoch_forward, val_epoch_backward, val_summ = get_epoch_loss(val_models, val_pair_indices)
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