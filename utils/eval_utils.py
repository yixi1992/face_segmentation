import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
import scipy
import os
import caffe
import random
import lmdb
import caffe.proto.caffe_pb2
from caffe.io import datum_to_array
from convert_to_lmdb import LoadImage, LoadLabel, ImageResizer, LabelResizer
import sys

class CArgs(object):
	pass

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


# Load Image data to a dictionary
def LoadImagePaths(image_path):
	#TODO maybe change convert_to_lmdb_flow.py to modules
	D = dict([(os.path.splitext(os.path.basename(x))[0], x) for x in sorted(glob.glob( image_path.format(id='*')))])
	return D

# Load Label Image data to a dictionary
def LoadLabelPaths(label_path):
	D = dict([(os.path.splitext(os.path.basename(x))[0].replace('_gt',''), x) for x in sorted(glob.glob( label_path.format(id='*')))]) 
	return D



# Mean Intersection over Union
# http://www.cs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
# (1/ncl) \sum_i n_ii / (\sum_j n_ij + \sum_j n_ji - nii)
def confcount(confcounts, pr, gt, args):
	for i in range(0, args.numclasses):
		for j in range(0, args.numclasses):
			confcounts[i][j] = confcounts[i][j] + np.sum((gt==i) & (pr==j)) 
	return confcounts

def acc_accumulator(class_acc, pr, gt, args):
	for i in range(0, args.numclasses):
		class_acc[i+args.numclasses] += np.sum(gt==i)
		class_acc[i] += np.sum((gt==i) & (pr==i))
	return class_acc


# Predict segmentation mask and return accuracy of a model_file on provided image/label set
def test_accuracy(model_file, image_dict, flows_dict, label_dict, args):
	print '-----------', 'model_file: ', model_file, '-----------'
	acc = []
	class_acc = np.zeros(args.numclasses*2)
	confcounts = np.zeros((args.numclasses, args.numclasses))
	resizer = None if not args.resize else ImageResizer(args.RSize, args.BoxSize, args.nopadding, args.RGB_pad_values, args.flow_pad_value)
	labelresizer = None if not args.resize else LabelResizer(args.LabelSize, args.BoxSize, args.nopadding, label_pad_value=args.BackGroundLabel)

	# load net
	if args.use_gpu:
		caffe.set_mode_gpu()
		caffe.set_device(0)
	net = caffe.Net(args.deploy_file, model_file, caffe.TEST)
	print len(image_dict.keys())
	
	for key in image_dict.keys():
		print key
		if random.random()>=args.test_ratio:
			continue
		img_path = image_dict[key]
		label_path = label_dict[key]
		flows_path = [flow_dict[key] for flow_dict in flows_dict]
		
		save_path = os.path.join(args.pred_visual_dir, '{version}_{idx}.png'.format(version=args.dataset_name, idx=key))
		if ((not args.shortcut_inference) or (not os.path.exists(save_path))):
			im = LoadImage(img_path, flows_path, resizer) 
			in_ = im.astype(np.float32)
			in_ = in_.transpose((1,2,0))
			# subtract mean from RGB
			print in_.shape, args.RGB_mean_values+len(flows_dict)*[128]
			in_ -= np.array(args.RGB_mean_values+len(flows_dict)*[128], dtype=np.float32)
			in_ = in_.transpose((2,0,1))

			# shape for input (data blob is N x C x H x W), set data
			net.blobs['data'].reshape(1, *in_.shape)
			net.blobs['data'].data[...] = in_
			# run net and take argmax for prediction
			net.forward()
			out = net.blobs['score'].data[0].argmax(axis=0)
			out = np.array(out, dtype=np.uint8)
			
			scipy.misc.imsave(save_path, out)
		else:
			out = scipy.misc.imread(save_path)
		
		L = np.array(LoadLabel(label_path), dtype=np.uint8)
		L = L.reshape(L.shape[1], L.shape[2])
		out = labelresizer.upsample(out, L.shape)
		out = np.array(out, dtype=np.uint8)

		if args.eval_metric=='pixel_accuracy':
			acc += [np.mean(out==L)]
			print('{version}_{idx} acc={acc} totalmatch={sum_pixel}'.format(version=v, idx=key, acc=np.mean(out==L), sum_pixel = np.sum(out==L)))
		elif args.eval_metric=='class_accuracy':
			# sum of class accuracy
			class_acc = acc_accumulator(class_acc, out, L, args)
		elif args.eval_metric=='eval_miu':
			# mIU intersection over union
			confcounts = confcount(confcounts, out, L, args)
	
	if args.eval_metric=='pixel_accuracy':
		print '-----------', 'model_file: ', model_file, '  acc:', np.mean(acc), '-----------'
		return(np.mean(acc))
	elif args.eval_metric=='class_accuracy':
		print '-----------', 'model_file: ', model_file
		mean_class_acc = np.divide(class_acc[:args.numclasses], class_acc[args.numclasses:])
		for i in range(0, args.numclasses):
			print 'class ', i, 'acc:', mean_class_acc[i]
		print '------------------------'
		return(np.mean(mean_class_acc))
	elif args.eval_metric=='eval_miu':
		# mIU intersection over union
		miu = 0
		miu_cnt = 0
		colsum = np.sum(confcounts, 0)
		rowsum = np.sum(confcounts, 1)
		for i in range(0, args.numclasses):
			if not (i in args.interested_class):
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
def plot_acc(x, y, v, eval_metric, work_dir):
	plt.clf()
	plt.plot(x,y, 'r*-')
	plt.ylabel(eval_metric)
	plt.title('{version} {metric}'.format(version=v, metric=eval_metric))
	plt.savefig(os.path.join(work_dir, '{version}_{metric}.png'.format(version=v, metric=eval_metric)))

# Main procedure which takes image/label sets and evaluate on a range of caffe models
def eval(inputs, inputs_flows, inputs_Label, args):
	acc = []
	for idx, snapshot_id in enumerate(args.iter): 
		model_file = args.snapshot.format(snapshot_id=snapshot_id)
		print(model_file)
		args.pred_visual_dir = args.pred_visual_dir_template.format(snapshot_id=snapshot_id)
		if not os.path.exists(args.pred_visual_dir):
			os.makedirs(args.pred_visual_dir)
		
		acc = acc + [test_accuracy(model_file, inputs, inputs_flows, inputs_Label, args)]
		plot_acc(args.iter[:(idx+1)], acc, args.dataset_name + '_' + args.model, args.eval_metric, args.work_dir)


