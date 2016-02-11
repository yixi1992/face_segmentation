import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
import scipy
import os

import caffe

Rheight=100
Rwidth=100
LabelHeight=100
LabelWidth=100

dataset='Train'
#dataset='Test'
image_dir='/lustre/yixi/data/massimomauro-FASSEG-dataset-f93e332/V2/'+dataset+'_RGB/*.bmp'
image_files = sorted(glob.glob(image_dir))
label_dir='/lustre/yixi/data/massimomauro-FASSEG-dataset-f93e332/V2/'+dataset+'_Labels/labels/'
label_suffix = '.png'

def test_accuracy(model_file):
	acc = np.zeros(len(image_files))
	for idx,image_file in enumerate(image_files):
		# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
		print image_file
		im = np.array(Image.open(image_file))
		im = im[:,:,::-1]
		im = Image.fromarray(im)
		im = im.resize([Rheight, Rwidth], Image.ANTIALIAS)
		#
		im = np.array(im, dtype=np.float32)
		in_ = im
		in_ -= np.array((0,0,0))
		in_ = in_.transpose((2,0,1))
		#
		# load net
		net = caffe.Net('face_segmentation_finetune_deploy.prototxt', model_file, caffe.TEST)
		# shape for input (data blob is N x C x H x W), set data
		net.blobs['data'].reshape(1, *in_.shape)
		net.blobs['data'].data[...] = in_
		# run net and take argmax for prediction
		net.forward()
		out = net.blobs['score'].data[0].argmax(axis=0)
		#
		scipy.misc.imsave('pred_visual/'+model_file[len('snapshots/snapshot_face_segmentation_finetune_'):(len(model_file)-len('.caffemodel.h5'))] + '/pred_' + image_file[len('/lustre/yixi/data/massimomauro-FASSEG-dataset-f93e332/V2/'+dataset+'_RGB/'):(len(image_file)-len('.bmp'))]+ '.png', out)
		#
		L = np.array(Image.open(label_dir + image_file[len('/lustre/yixi/data/massimomauro-FASSEG-dataset-f93e332/V2/'+dataset+'_RGB/'):(len(image_file)-len('.bmp'))] + label_suffix)) # or load whatever ndarray you need
		Dtype = L.dtype
		Limg = Image.fromarray(L)
		Limg = Limg.resize([LabelHeight, LabelWidth],Image.NEAREST) # To resize the Label file to the required size 
		L = np.array(Limg,Dtype)
		L = L.reshape(L.shape[0],L.shape[1],1)
		L = L.transpose((2,0,1))
		#
		acc[idx] = np.mean(out==L)
		print(str(idx), ': acc=', np.mean(out==L))
	return(np.mean(acc))



iter = range(62970, 62975, 5)
model_acc = np.zeros(len(iter))
for idx,i in enumerate(iter):
	model_file = 'snapshots/snapshot_face_segmentation_finetune_fixlr1e-8__iter_'+str(i)+'.caffemodel.h5'
	if not os.path.exists('pred_visual/'+model_file[len('snapshots/snapshot_face_segmentation_finetune_'):len('snapshots/snapshot_face_segmentation_finetune_fixlr1e-8__iter_'+str(i))] + '/'):
		os.makedirs('pred_visual/'+model_file[len('snapshots/snapshot_face_segmentation_finetune_'):len('snapshots/snapshot_face_segmentation_finetune_fixlr1e-8__iter_'+str(i))] + '/')
	model_acc[idx] = test_accuracy(model_file)
	#
	plt.clf()
	plt.plot(model_acc[:(idx+1)])
	plt.ylabel('accuracy')
	plt.title('accuracy on '+dataset)
	plt.savefig(dataset+'_accuracy_fixlr1e-8.png')
	

