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

# Experiment Details
parser.add_argument('--exp', type=str, required=True, help='[Required] Path of experiment for loading pre-trained model')
parser.add_argument('--category', type=str, required=True, help='[Required] Model Category for training')
parser.add_argument('--load_best', action='store_true', help='load best val model')

#parser.add_argument('--bn_encoder', action='store_true', help='Supply this parameter if you want bn_encoder, otherwise ignore')
parser.add_argument('--bn_decoder', action='store_true', help='Supply this parameter if you want bn_decoder, otherwise ignore')
parser.add_argument('--bn_decoder_final', action='store_true', help='Supply this parameter if you want bn_decoder, otherwise ignore')

# Fetch Batch Details
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during evaluation')
parser.add_argument('--eval_set', type=str, help='Choose from train/valid')

# Other Args
parser.add_argument('--visualize', action='store_true', help='supply this parameter to visualize')

FLAGS = parser.parse_args()

print '-='*50
print FLAGS
print '-='*50

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

BATCH_SIZE = FLAGS.batch_size
if FLAGS.visualize:
	BATCH_SIZE = 1
NUM_POINTS = 2048
NUM_EVAL_POINTS = 2048
IN_PCL_SIZE = 1024
NUM_VIEWS = 24
HEIGHT = 128
WIDTH = 128
PAD = 35

if FLAGS.visualize:
	from utils.show_3d import show3d_balls
	ballradius = 3

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

	ph_gt = tf.placeholder(tf.float32, (BATCH_SIZE, NUM_POINTS, 3), name='ph_gt')
	ph_pr = tf.placeholder(tf.float32, (BATCH_SIZE, IN_PCL_SIZE, 3), name='ph_pr')
    img_inp = tf.placeholder(tf.float32, shape=(BATCH_SIZE, HEIGHT, WIDTH, 3), name='img_inp')

	dists_forward, dists_backward, chamfer_distance = get_chamfer_metrics(ph_gt, ph_pr)
	emd = get_emd_metrics(ph_gt, ph_pr, BATCH_SIZE, NUM_POINT)

	for cnt in iters:
		start = time.time()

		batch_ip, batch_gt = fetch_batch_shapenet(models, indices, cnt, BATCH_SIZE)

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
	img_inp = tf.placeholder(tf.float32, shape=(BATCH_SIZE, HEIGHT, WIDTH, 3), name='img_inp')
	pcl_gt = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_EVAL_POINTS, 3), name='pcl_gt')
