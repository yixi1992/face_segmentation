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


	
if False:
	lmdb_dir = 'mass_lmdb'

if True:
	lmdb_dir = 'camvid_lmdb'
	work_dir = '/lustre/yixi/face_segmentation_finetune/fullconv'
	deploy_file = os.path.join(work_dir, 'deploy.prototxt')
	snapshot = os.path.join(work_dir, 'snapshots_camvid/train_lr1e-12/_iter_{snapshot_id}.caffemodel')
	pred_visual_dir_template = os.path.join(work_dir, 'pred_visual_camvid/train_lr1e-12/_iter_{snapshot_id}')
	iter = range(200, 201, 200)



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

inputs_Train = LMDB2Dict(os.path.join(lmdb_dir,'train-lmdb'))
inputs_Train_Label = LMDB2Dict(os.path.join(lmdb_dir,'train-label-lmdb'))
inputs_Test = LMDB2Dict(os.path.join(lmdb_dir,'test-lmdb'))
inputs_Test_Label = LMDB2Dict(os.path.join(lmdb_dir,'test-label-lmdb'))





def test_accuracy(model_file, image_dict, label_dict, pred_visual_dir, v):
	acc = np.zeros(len(image_dict))
	for  in_idx, in_ in image_dict.iteritems():
		# load net
		net = caffe.Net(deploy_file, model_file, caffe.TEST)
		# shape for input (data blob is N x C x H x W), set data
		net.blobs['data'].reshape(1, *in_.shape)
		net.blobs['data'].data[...] = in_
		# run net and take argmax for prediction
		net.forward()
		out = net.blobs['score'].data[0].argmax(axis=0)
		out=np.array(out, dtype=np.uint8)
		
		scipy.misc.imsave(os.path.join(pred_visual_dir, '{version}_{idx}.png'.format(version=v, idx=in_idx)), out)
		
		L = label_dict[in_idx]
		
		acc[in_idx] = np.mean(out==L)
		print('{version}_{idx} acc={acc}'.format(version=v, idx=in_idx, acc=np.mean(out==L)))
	return(np.mean(acc))


def plot_acc(x, y, v):
	plt.clf()
	plt.plot(x,y)
	plt.ylabel('accuracy')
	plt.title('{version} accuracy'.format(version=v))
	plt.savefig(os.path.join(work_dir, '{version}_accuracy.png'.format(version=v)))


train_acc = np.zeros(len(iter))
test_acc = np.zeros(len(iter))
for idx,snapshot_id in enumerate(iter): 
	model_file = snapshot.format(snapshot_id=snapshot_id)
	print(model_file)
	pred_visual_dir = pred_visual_dir_template.format(snapshot_id=snapshot_id)
	if not os.path.exists(pred_visual_dir):
		os.makedirs(pred_visual_dir)
	
	train_acc[idx] = test_accuracy(model_file, inputs_Train, inputs_Train_Label, pred_visual_dir, 'Train')
	plot_acc(iter[:(idx+1)], train_acc[:(idx+1)], 'Train')
	
	test_acc[idx] = test_accuracy(model_file, inputs_Test, inputs_Test_Label, pred_visual_dir, 'Test')
	plot_acc(iter[:(idx+1)], test_acc[:(idx+1)], 'Test')
