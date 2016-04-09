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
from convert_to_lmdb_flow import LoadImage, LoadLabel, ImageResizer, LabelResizer



# Load Image data to a dictionary
def LoadImagePaths(image_path):
	#TODO maybe change convert_to_lmdb_flow.py to modules
	D = dict([(os.path.splitext(os.path.basename(x))[0], x) for x in sorted(glob.glob( image_path.format(id='*')))])
	return D

# Load Label Image data to a dictionary
def LoadLabelPaths(label_path):
	D = dict([(os.path.splitext(os.path.basename(x))[0].replace('_gt',''), x) for x in sorted(glob.glob( label_path.format(id='*')))]) 
	return D

def LoadFlowPaths(flow_path, keys=None):
	if keys!=None:		
		D = dict([(id, flow_path.format(id=id)) for id in keys]) 
	else:
		D = dict([(os.path.splitext(os.path.basename(x))[0].replace('.f1.flow_x','').replace('.f1.flow_y', ''), x) for x in sorted(glob.glob( flow_path.format(id='*')))]) 
	return D



# Mean Intersection over Union
# http://www.cs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
# (1/ncl) \sum_i n_ii / (\sum_j n_ij + \sum_j n_ji - nii)
def confcount(confcounts, pr, gt):
	for i in range(0, numclasses):
		for j in range(0, numclasses):
			confcounts[i][j] = confcounts[i][j] + np.sum((gt==i) & (pr==j)) 
	return confcounts

def acc_accumulator(class_acc, pr, gt):
	for i in range(0, numclasses):
		class_acc[i+numclasses] += np.sum(gt==i)
		class_acc[i] += np.sum((gt==i) & (pr==i))
	return class_acc


# Predict segmentation mask and return accuracy of a model_file on provided image/label set
def test_accuracy(model_file, image_dict, label_dict, flow_x_dict, flow_y_dict, pred_visual_dir, v):
	print '-----------', 'model_file: ', model_file, '-----------'
	acc = []
	class_acc = np.zeros(numclasses*2)
	confcounts = np.zeros((numclasses, numclasses))
	resizer = None if not resize else ImageResizer(RSize, BoxSize)
	labelresizer = None if not resize else LabelResizer(LabelSize, BoxSize)

	# load net
	net = caffe.Net(deploy_file, model_file, caffe.TEST)
	print len(image_dict.keys())
	
	for key in image_dict.keys():
		print key
		img_path = image_dict[key]
		label_path = label_dict[key]
		flow_x_path = flow_x_dict[key] if flow_x_dict!=None and (key in flow_x_dict.keys()) else None
		flow_y_path = flow_y_dict[key] if flow_y_dict!=None and (key in flow_y_dict.keys()) else None
		
		save_path = os.path.join(pred_visual_dir, '{version}_{idx}.png'.format(version=v, idx=key))
		if ((not shortcut_inference) or (not os.path.exists(save_path))):
			im = LoadImage(img_path, flow_x_path, flow_y_path, resizer) 
			in_ = im.astype(np.float32)
			in_ = in_.transpose((1,2,0))
			# subtract mean from RGB
			in_ -= np.array(input_RGB_mean[v], dtype=np.float32)
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

		if eval_metric=='pixel_accuracy':
			acc += [np.mean(out==L)]
			print('{version}_{idx} acc={acc} totalmatch={sum_pixel}'.format(version=v, idx=key, acc=np.mean(out==L), sum_pixel = np.sum(out==L)))
		elif eval_metric=='class_accuracy':
			# sum of class accuracy
			class_acc = acc_accumulator(class_acc, out, L)
		elif eval_metric=='eval_miu':
			# mIU intersection over union
			confcounts = confcount(confcounts, out, L)
	
	if eval_metric=='pixel_accuracy':
		print '-----------', 'model_file: ', model_file, '  acc:', np.mean(acc), '-----------'
		return(np.mean(acc))
	elif eval_metric=='class_accuracy':
		print '-----------', 'model_file: ', model_file
		mean_class_acc = np.divide(class_acc[:numclasses], class_acc[numclasses:])
		for i in range(0, numclasses):
			print 'class ', i, 'acc:', mean_class_acc[i]
		print '------------------------'
		return(np.mean(mean_class_acc))
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
	plt.plot(x,y, 'r*-')
	plt.ylabel(eval_metric)
	plt.title('{version} {metric}'.format(version=v, metric=eval_metric))
	plt.savefig(os.path.join(work_dir, '{version}_{metric}.png'.format(version=v, metric=eval_metric)))

# Main procedure which takes image/label sets and evaluate on a range of caffe models
def eval(inputs, inputs_Label, inputs_flow_x, inputs_flow_y, dataset):
	acc = []
	for idx,snapshot_id in enumerate(iter): 
		model_file = snapshot.format(snapshot_id=snapshot_id)
		print(model_file)
		pred_visual_dir = pred_visual_dir_template.format(snapshot_id=snapshot_id)
		if not os.path.exists(pred_visual_dir):
			os.makedirs(pred_visual_dir)
		
		acc = acc + [test_accuracy(model_file, inputs, inputs_Label, inputs_flow_x, inputs_flow_y, pred_visual_dir, dataset)]
		plot_acc(iter[:(idx+1)], acc, dataset + '_' + model)

if __name__=='__main__':

	resize = True
	RSize = (200, 200)
	LabelSize = (200, 200)
	nopadding = False
	useflow = False
	
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
		model = 'modeldefault_snapshots_gcfvshuffle200200_train_lr1e-10'
		work_dir = '/lustre/yixi/face_segmentation_finetune/gcfv_flow/modeldefault'
		deploy_file = os.path.join(work_dir, 'deploy.prototxt')
		snapshot = os.path.join(work_dir, 'snapshots_gcfvshuffle200200/train_lr1e-10/_iter_{snapshot_id}.caffemodel')
		pred_visual_dir_template = os.path.join(work_dir, 'pred_visual_gcfvshuffle200200/train_lr1e-10/_iter_{snapshot_id}')
		
		train_data = '/lustre/yixi/data/gcfv_dataset/cross_validation/videos/frames/{id}.jpg'
		train_label_data = '/lustre/yixi/data/gcfv_dataset/cross_validation/ground_truth/labels/{id}_gt.png'
		test_data = '/lustre/yixi/data/gcfv_dataset/external_validation/videos/frames/{id}.jpg'
		test_label_data = '/lustre/yixi/data/gcfv_dataset/external_validation/ground_truth/labels/{id}_gt.png'
		train_flow_x = '/lustre/yixi/data/gcfv_dataset/cross_validation/videos/flow/{id}.f1.flow_x.png'
		train_flow_y = '/lustre/yixi/data/gcfv_dataset/cross_validation/videos/flow/{id}.f1.flow_y.png'
		test_flow_x = '/lustre/yixi/data/gcfv_dataset/external_validation/videos/flow/{id}.f1.flow_x.png'
		test_flow_y = '/lustre/yixi/data/gcfv_dataset/external_validation/videos/flow/{id}.f1.flow_y.png'

		iter = range(30000, 25000, -5000)
		
		BoxSize = None
			
		numclasses = 9
		interested_class = range(0, numclasses)
		
		eval_metric = 'pixel_accuracy'
		input_RGB_mean = {'Train':(83.3774396271, 87.3343435075, 86.27596998),
				'Test':(83.3774396271, 87.3343435075, 86.27596998)}
		shortcut_inference = True




	inputs_Test = LoadImagePaths(test_data)
	inputs_Test_Label = LoadLabelPaths(test_label_data)
	flow_x_Test = None if not useflow else dict([(id, test_flow_x.format(id=id)) for id in inputs_Test.keys()])
	flow_y_Test = None if not useflow else dict([(id, test_flow_y.format(id=id)) for id in inputs_Test.keys()])
	eval(inputs_Test, inputs_Test_Label, flow_x_Test, flow_y_Test, 'Test')
	inputs_Test.clear()
	inputs_Test_Label.clear()

	inputs_Train = LoadImagePaths(train_data)
	inputs_Train_Label = LoadLabelPaths(train_label_data)
	flow_x_Train = None if not useflow else dict([(id, train_flow_x.format(id=id)) for id in inputs_Train.keys()])
	flow_y_Train = None if not useflow else dict([(id, train_flow_y.format(id=id)) for id in inputs_Train.keys()])
	#eval(inputs_Train, inputs_Train_Label, flow_x_Train, flow_y_Train, 'Train')
	inputs_Train.clear()
	inputs_Train_Label.clear()

