import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
import scipy
import os
import caffe
import lmdb
import caffe.proto.caffe_pb2
from caffe.io import datum_to_array

# Load LMDB data to a dictionary
def LMDB2Dict(lmdb_directory):
	D = dict()
	lmdb_env = lmdb.open(lmdb_directory)
	lmdb_txn = lmdb_env.begin()
	lmdb_cursor = lmdb_txn.cursor()
	datum = caffe.proto.caffe_pb2.Datum()
	for key, value in lmdb_cursor:
		datum.ParseFromString(value)
		data = caffe.io.datum_to_array(datum)
		D[key] = data
	return D

# Mean Intersection over Union
# http://www.cs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
# (1/ncl) \sum_i n_ii / (\sum_j n_ij + \sum_j n_ji - nii)
def confcount(confcounts, pr, gt):
	for i in range(0, numclasses):
		for j in range(0, numclasses):
			confcounts[i][j] = confcounts[i][j] + np.sum((gt==i) & (pr==j)) 
	return confcounts

# Predict segmentation mask and return accuracy of a model_file on provided image/label set
def test_accuracy(model_file, image_dict, label_dict, pred_visual_dir, v):
	print '-----------', 'model_file: ', model_file, '-----------'
	acc = []
	confcounts = np.zeros((numclasses, numclasses))
	for in_idx, in_ in image_dict.iteritems():	
		file_path = os.path.join(pred_visual_dir, '{version}_{idx}.png'.format(version=v, idx=in_idx))
		if ((not shortcut_inference) or (not os.path.exists(file_path))):
			in_ = in_.astype(np.float32)
			# subtract mean from RGB
			in_ = in_.transpose((1,2,0))
			in_ -= np.array(input_RGB_mean[v], dtype=np.float32)
			in_ = in_.transpose((2,0,1))

			# load net
			net = caffe.Net(deploy_file, model_file, caffe.TEST)
			# shape for input (data blob is N x C x H x W), set data
			net.blobs['data'].reshape(1, *in_.shape)
			net.blobs['data'].data[...] = in_
			# run net and take argmax for prediction
			net.forward()
			out = net.blobs['score'].data[0].argmax(axis=0)
			out = np.array(out, dtype=np.uint8)
			
			scipy.misc.imsave(file_path, out)
		else:
			out = scipy.misc.imread(file_path)
		
		L = label_dict[in_idx]
		
		if eval_metric=='pixel_accuracy':
			# pixel accuracy
			acc = acc + [np.mean(out==L)]
			print('{version}_{idx} acc={acc}'.format(version=v, idx=in_idx, acc=np.mean(out==L)))
		elif eval_metric=='eval_miu':
			# mIU intersection over union
			confcounts = confcount(confcounts, out, L)
	
	if eval_metric=='pixel_accuracy':
		print '-----------', 'model_file: ', model_file, '  acc:', np.mean(acc), '-----------'
		return(np.mean(acc))	
	elif eval_metric=='eval_miu':
		# mIU intersection over union
		miu = 0
		miu_cnt = 0
		colsum = np.sum(confcounts, 0)
		rowsum = np.sum(confcounts, 1)
		for i in range(0, numclasses):
			if not (i in interested_class):
				continue
			if (rowsum[i]+colsum[i]-confcounts[i][i]>1e-6):
				miu2 = float(confcounts[i][i])/float(rowsum[i]+colsum[i]-confcounts[i][i])
				miu_cnt += 1
			else:
				miu2 = 0
				continue
			miu = miu + miu2
			print i, ' miu=', miu/miu_cnt, '    ', miu2, '=', confcounts[i][i], '/', rowsum[i], ',', colsum[i], '-----------'
		mmiu = miu/miu_cnt
		print '-----------', 'model_file: ', model_file, '  miu:', mmiu, '-----------'
		return mmiu


# Plot the accuracy curve and save to file
def plot_acc(x, y, v):
	plt.clf()
	plt.plot(x,y)
	plt.ylabel(eval_metric)
	plt.title('{version} {metric}'.format(version=v, metric=eval_metric))
	plt.savefig(os.path.join(work_dir, '{version}_{metric}.png'.format(version=v, metric=eval_metric)))

# Main procedure which takes image/label sets and evaluate on a range of caffe models
def eval(inputs, inputs_Label, dataset):
	acc = []
	for idx,snapshot_id in enumerate(iter): 
		model_file = snapshot.format(snapshot_id=snapshot_id)
		print(model_file)
		pred_visual_dir = pred_visual_dir_template.format(snapshot_id=snapshot_id)
		if not os.path.exists(pred_visual_dir):
			os.makedirs(pred_visual_dir)
		
		acc = acc + [test_accuracy(model_file, inputs, inputs_Label, pred_visual_dir, dataset)]
		plot_acc(iter[:(idx+1)], acc, dataset + '_' + model)

#MODIFY ME
if False:
	# NOT tested
	lmdb_dir = 'mass_lmdb'

if False:
	model = 'snapshots_camvid200200_train_lr1e-10'
	lmdb_dir = '../camvid200200_lmdb'
	work_dir = '/lustre/yixi/face_segmentation_finetune/flow/modeldefault'
	deploy_file = os.path.join(work_dir, 'deploy.prototxt')
	snapshot = os.path.join(work_dir, 'snapshots_camvid200200/train_lr1e-10/_iter_{snapshot_id}.caffemodel')
	pred_visual_dir_template = os.path.join(work_dir, 'pred_visual_camvid200200/train_lr1e-10/_iter_{snapshot_id}')
	iter = range(77000, 74000, -1000)
	numclasses = 33
	#interested_class = range(0, numclasses)
	interested_class = [2, 4, 5, 8, 9, 16, 17, 19, 20, 21, 26]
	eval_metric = 'eval_miu'
	input_RGB_mean = {'Train':(78.1049806452, 76.5400646313, 74.0029471198),
			'Test':(78.1049806452, 76.5400646313, 74.0029471198)}
	shortcut_inference = True

if True:
	model = 'snapshots_gcfv200200_train_lr1e-10'
	lmdb_dir = '../camvid200200_lmdb'
	work_dir = '/lustre/yixi/face_segmentation_finetune/gcfv_flow/modeldefault'
	deploy_file = os.path.join(work_dir, 'deploy.prototxt')
	snapshot = os.path.join(work_dir, 'snapshots_gcfv200200/train_lr1e-10/_iter_{snapshot_id}.caffemodel')
	pred_visual_dir_template = os.path.join(work_dir, 'pred_visual_gcfv200200/train_lr1e-10/_iter_{snapshot_id}')
	iter = range(77000, 74000, -1000)
	numclasses = 9
	interested_class = range(0, numclasses)
	eval_metric = 'eval_miu'
	input_RGB_mean = {'Train':(83.3774396271, 87.3343435075, 86.27596998),
			'Test':(83.3774396271, 87.3343435075, 86.27596998)}
	shortcut_inference = True




inputs_Test = LMDB2Dict(os.path.join(lmdb_dir,'test-lmdb'))
inputs_Test_Label = LMDB2Dict(os.path.join(lmdb_dir,'test-label-lmdb'))
eval(inputs_Test, inputs_Test_Label, 'Test')
inputs_Test.clear()
inputs_Test_Label.clear()


inputs_Train = LMDB2Dict(os.path.join(lmdb_dir,'train-lmdb'))
inputs_Train_Label = LMDB2Dict(os.path.join(lmdb_dir,'train-label-lmdb'))
eval(inputs_Train, inputs_Train_Label, 'Train')
inputs_Train.clear()
inputs_Train_Label.clear()


