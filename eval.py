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
def mIU(pr, gt):
	R = max(np.amax(pr), np.amax(gt))
	L = min(np.amin(pr), np.amin(gt))
	t = np.zeros(R+1) #sum_j n_ij
	s = np.zeros(R+1) #sum_j n_ji
	n = np.zeros(R+1)
	acc = []
	for i in range(L,R+1,1):
		t[i] = np.sum(pr==i)
		s[i] = np.sum(gt==i)
		n[i] = np.sum((pr==i) & (gt==i))
		if not (t[i]+s[i]-n[i]==0):
			acc = acc + n[i]/(t[i]+s[i]-n[i])
	return np.mean(acc)

# Predict segmentation mask and return accuracy of a model_file on provided image/label set
def test_accuracy(model_file, image_dict, label_dict, pred_visual_dir, v):
	acc = []
	# load net
	net = caffe.Net(deploy_file, model_file, caffe.TEST)
	for in_idx, in_ in image_dict.iteritems():	
		# shape for input (data blob is N x C x H x W), set data
		net.blobs['data'].reshape(1, *in_.shape)
		net.blobs['data'].data[...] = in_
		# run net and take argmax for prediction
		net.forward()
		out = net.blobs['score'].data[0].argmax(axis=0)
		out = np.array(out, dtype=np.uint8)
		
		scipy.misc.imsave(os.path.join(pred_visual_dir, '{version}_{idx}.png'.format(version=v, idx=in_idx)), out)
		
		L = label_dict[in_idx]
		
		#mIU intersection over union
		acc = acc + mIU(out, L)
		#pixel accuracy
		#acc = acc + [np.mean(out==L)]
		print('{version}_{idx} acc={acc}'.format(version=v, idx=in_idx, acc=np.mean(out==L)))
	return(np.mean(acc))

# Plot the accuracy curve and save to file
def plot_acc(x, y, v):
	plt.clf()
	plt.plot(x,y)
	plt.ylabel('mIU')
	plt.title('{version} mIU'.format(version=v))
	plt.savefig(os.path.join(work_dir, '{version}_accuracy.png'.format(version=v)))

# Main procedure which takes image/label sets and evaluate on a range of caffe models
def eval(inputs, inputs_Label, dataset):
	acc = np.zeros(len(iter))
	for idx,snapshot_id in enumerate(iter): 
		model_file = snapshot.format(snapshot_id=snapshot_id)
		print(model_file)
		pred_visual_dir = pred_visual_dir_template.format(snapshot_id=snapshot_id)
		if not os.path.exists(pred_visual_dir):
			os.makedirs(pred_visual_dir)
		
		acc[idx] = test_accuracy(model_file, inputs, inputs_Label, pred_visual_dir, dataset)
		plot_acc(iter[:(idx+1)], acc[:(idx+1)], dataset + '_' + model)


#MODIFY ME
if False:
	lmdb_dir = 'mass_lmdb'

if True:
	model = 'camvid_lr1e-12'
	lmdb_dir = 'camvid_lmdb'
	work_dir = '/lustre/yixi/face_segmentation_finetune/fullconv'
	deploy_file = os.path.join(work_dir, 'deploy.prototxt')
	snapshot = os.path.join(work_dir, 'snapshots_camvid/train_lr1e-12/_iter_{snapshot_id}.caffemodel')
	pred_visual_dir_template = os.path.join(work_dir, 'pred_visual_camvid/train_lr1e-12/_iter_{snapshot_id}')
	iter = range(200, 4001, 200)


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