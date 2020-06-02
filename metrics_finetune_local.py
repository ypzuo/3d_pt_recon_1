from importer import *
from utils.icp import icp
from tqdm import tqdm
import utils.pointnet2_utils.tf_util
from utils.pointnet2_utils.pointnet_util import pointnet_sa_module, pointnet_fp_module, pointnet_sa_module_msg
from utils.pointnet2_utils.pointSIFT_util import pointSIFT_module
parser = argparse.ArgumentParser()

# Machine Details
parser.add_argument('--gpu', type=str, required=True, help='[Required] GPU to use')

# Dataset
parser.add_argument('--data_dir_imgs', type=str, required=True, help='Path to shapenet rendered images')
parser.add_argument('--data_dir_pcl', type=str, required=True, help='Path to shapenet pointclouds')
# parser.add_argument('--data_dir', type=str, required=True, help='Path to shapenet rendered images')

# Experiment Details
parser.add_argument('--exp', type=str, required=True, help='[Required] Path of experiment for loading pre-trained model')
parser.add_argument('--ae_logs', type=str, help='Location of pretrained auto-encoder snapshot')
parser.add_argument('--category', type=str, required=True, help='[Required] Model Category for training')
parser.add_argument('--load_best', action='store_true', help='load best val model')

# AE Details
parser.add_argument('--bottleneck', type=int, required=False, default=512, help='latent space size')
#parser.add_argument('--load_best_ae', action='store_true', help='supply this parameter to load best model from the auto-encoder')
# parser.add_argument('--bn_encoder', action='store_true', help='Supply this parameter if you want bn_encoder, otherwise ignore')
parser.add_argument('--bn_decoder', action='store_true', help='Supply this parameter if you want bn_decoder, otherwise ignore')
parser.add_argument('--bn_decoder_final', action='store_true', help='Supply this parameter if you want bn_decoder, otherwise ignore')

# Fetch Batch Details
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during evaluation')
parser.add_argument('--eval_set', type=str, help='Choose from train/valid')

# Other Args
parser.add_argument('--visualize', action='store_true', help='supply this parameter to visualize')
parser.add_argument('--localsift', action='store_true', 
	help='Supply this parameter if you want to use localsift point feature, otherwise ignore')


FLAGS = parser.parse_args()

print '-='*50
print FLAGS
print '-='*50

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

BATCH_SIZE = FLAGS.batch_size
if FLAGS.visualize:
	BATCH_SIZE = 1
NUM_POINTS = 2048
NUM_EVAL_POINTS = 1024
OUT_PCL_SIZE = 2048
IN_PCL_SIZE = 1024
NUM_VIEWS = 24
UPSAMPLING_FACTOR = 2
HEIGHT = 128
WIDTH = 128
PAD = 35

if FLAGS.visualize:
	from utils.show_3d import show3d_balls
	ballradius = 3


def fetch_batch_shapenet(models, indices, batch_num, batch_size):
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
		Batch Loader for ShapeNet dataset
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

def fetch_batch_dense(models, indices, batch_num, batch_size):
	batch_ip = []
	batch_gt = []
	batch_img = []
	for ind in indices[batch_num*batch_size:batch_num*batch_size+batch_size]:
		model_path = models[ind[0]]
		img_path = join(FLAGS.data_dir_imgs, model_path, 'rendering', PNG_FILES[ind[1]])
		pcl_gt = np.load(join(FLAGS.data_dir_pcl, model_path, 'pointcloud_2048.npy'))
		pcl_ip = np.load(join(FLAGS.data_dir_pcl, model_path,'pointcloud_1024.npy'))
		ip_image = cv2.imread(img_path)[4:-5, 4:-5, :3]
		ip_image = cv2.cvtColor(ip_image, cv2.COLOR_BGR2RGB)
		batch_gt.append(pcl_gt)
		batch_ip.append(pcl_ip)
		batch_img.append(ip_image)
	batch_gt = np.array(batch_gt)
	batch_ip = np.array(batch_ip)
	batch_img = np.array(batch_img)
	return batch_ip, batch_gt, batch_img


def calculate_metrics(models, batches, pcl_gt_scaled, pred_scaled, indices=None):

	if FLAGS.visualize:
		iters = range(batches)
	else:
		iters = tqdm(range(batches))

	epoch_chamfer = 0.
	epoch_forward = 0.
	epoch_backward = 0.
	epoch_emd = 0.	

	ph_gt = tf.placeholder(tf.float32, (BATCH_SIZE, NUM_EVAL_POINTS, 3), name='ph_gt')
	ph_pr = tf.placeholder(tf.float32, (BATCH_SIZE, NUM_EVAL_POINTS, 3), name='ph_pr')

	dists_forward, dists_backward, chamfer_distance = get_chamfer_metrics(ph_gt, ph_pr)
	emd = get_emd_metrics(ph_gt, ph_pr, BATCH_SIZE, NUM_EVAL_POINTS)

	for cnt in iters:
		start = time.time()

		batch_ip, batch_gt, batch_img = fetch_batch_dense(models, indices, cnt, BATCH_SIZE)


		_gt_scaled, _pr_scaled = sess.run(
			[pcl_gt_scaled, pred_scaled], 
			feed_dict={pcl_in:batch_ip, pcl_gt:batch_gt, img_inp:batch_img}
		)

		_pr_scaled_icp = []

		for i in xrange(BATCH_SIZE):
			rand_indices = np.random.permutation(NUM_POINTS)[:NUM_EVAL_POINTS]
			T, _, _ = icp(_gt_scaled[i], _pr_scaled[i][rand_indices], tolerance=1e-10, max_iterations=1000)
			_pr_scaled_icp.append(np.matmul(_pr_scaled[i][rand_indices], T[:3,:3]) - T[:3, 3])

		_pr_scaled_icp = np.array(_pr_scaled_icp).astype('float32')

		C,F,B,E = sess.run(
			[chamfer_distance, dists_forward, dists_backward, emd], 
			feed_dict={ph_gt:_gt_scaled, ph_pr:_pr_scaled_icp}
		)

		epoch_chamfer += C.mean() / batches
		epoch_forward += F.mean() / batches
		epoch_backward += B.mean() / batches
		epoch_emd += E.mean() / batches

		if FLAGS.visualize:
			for i in xrange(BATCH_SIZE):
				print '-'*50
				print C[i], F[i], B[i], E[i]
				print '-'*50
				cv2.imshow('', batch_ip[i])

				print 'Displaying Gt scaled 1k'
				show3d_balls.showpoints(_gt_scaled[i], ballradius=3)
				print 'Displaying Pr scaled icp 1k'
				show3d_balls.showpoints(_pr_scaled_icp[i], ballradius=3)
		
		if cnt%10 == 0:
			print '%d / %d' % (cnt, batches)

	if not FLAGS.visualize:
		log_values(csv_path, epoch_chamfer, epoch_forward, epoch_backward, epoch_emd)

	return 


if __name__ == '__main__':

# Create Placeholders
	#pcl_gt = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINTS, 3))
	img_inp = tf.placeholder(tf.float32, shape=(BATCH_SIZE, HEIGHT, WIDTH, 3), name='img_inp')
	pcl_in = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IN_PCL_SIZE, 3), name='pcl_in')
	pcl_gt = tf.placeholder(tf.float32, shape=(BATCH_SIZE, OUT_PCL_SIZE, 3), name='pcl_gt')

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
		z = encoder_with_convs_and_symmetry(in_signal=reconstr_img, n_filters=[32,64,64], 
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
		print 'finnal out: ', out.shape
		reconstr_pcl = tf.reshape(out, (BATCH_SIZE, NUM_POINTS, 3))


	# Perform Scaling
	pcl_gt_scaled, reconstr_img_scaled = scale(pcl_gt, reconstr_pcl)

	#pointnet_ae_logs_path = FLAGS.ae_logs

	# Snapshot Folder Location
	if FLAGS.load_best:
		snapshot_folder = join(FLAGS.exp, 'best')
	else:
		snapshot_folder = join(FLAGS.exp, 'snapshots')

 	# Metrics path
 	metrics_folder = join(FLAGS.exp, 'metrics_shapenet', FLAGS.eval_set)
 	create_folder(metrics_folder)
	csv_path = join(metrics_folder,'%s.csv'%FLAGS.category)
	with open(csv_path, 'w') as f:
		f.write('Chamfer, Fwd, Bwd, Emd\n')

	# GPU configuration
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		# Load pretrained frozen pointnet ae weights
		#load_pointnet_ae(pointnet_ae_logs_path, pointnet_ae_vars, sess, FLAGS)

		saver = tf.train.Saver()
		load_previous_checkpoint(snapshot_folder, saver, sess, is_training=False)
		tflearn.is_training(False, session=sess)

		train_models, val_models, train_pair_indices, val_pair_indices = get_shapenet_models(FLAGS)
			
		if FLAGS.visualize:
			random.shuffle(val_pair_indices)
			random.shuffle(train_pair_indices)
			
		if FLAGS.eval_set == 'train':
			batches = len(train_pair_indices)
			calculate_metrics(train_models, batches, pcl_gt_scaled, reconstr_img_scaled, train_pair_indices)
		elif FLAGS.eval_set == 'valid':
			batches = len(val_pair_indices)
			calculate_metrics(val_models, batches, pcl_gt_scaled, reconstr_img_scaled, val_pair_indices)

		else:
			print 'Invalid dataset. Choose from [shapenet, pix3d]'
			sys.exit(1)
