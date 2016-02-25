import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
import scipy
import os

import caffe

Rheight = 224 # Required Height
Rwidth = 224 # Required Width
RheightLabel = 224 # Height for the label
RwidthLabel = 224 # Width for the label
LabelWidth = 224 # Downscaled width of the label
LabelHeight = 224 # Downscaled height of the label


dataset='valid'
inputs_label_train = sorted(glob.glob("/lustre/yixi/data/PASCAL-Context/trainval/*.mat"))
inputs_id_valid = [line.rstrip('\n') for line in open('/lustre/yixi/data/VOCdevkit/VOC2012/ImageSets/Main/val.txt')]
image_files = ['/lustre/yixi/data/VOCdevkit/VOC2012/JPEGImages/'+line[-15:-4]+'.jpg' for line in inputs_label_train if line[-15:-4] in set(inputs_id_valid)]
label_dir='/lustre/yixi/data/PASCAL-Context/trainval_PNG/'
label_suffix = '.PNG'

image_files = image_files[:3]


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
		in_ -= np.array((104.00698793,116.66876762,122.67891434))
		in_ = in_.transpose((2,0,1))
		#
		# load net
		net = caffe.Net('deploy.prototxt', model_file, caffe.TEST)
		# shape for input (data blob is N x C x H x W), set data
		net.blobs['data'].reshape(1, *in_.shape)
		net.blobs['data'].data[...] = in_
		# run net and take argmax for prediction
		net.forward()
		out = net.blobs['score'].data[0].argmax(axis=0)
		#
		scipy.misc.imsave('pred_visual/iter_'+model_file[len('snapshot/train_iter_'):(len(model_file)-len('.caffemodel'))] + '/pred_' + image_file[-15:-4]+ '.png', out)
		#
		L = np.array(Image.open(label_dir + image_file[-15:-4]+ label_suffix)) # or load whatever ndarray you need
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



iter = range(100, 3501,100)
model_acc = np.zeros(len(iter))
for idx,i in enumerate(iter):
	model_file = 'snapshot/train_iter_'+str(i)+'.caffemodel'
	if not os.path.exists('pred_visual/iter_'+model_file[len('snapshot/train_iter_'):len('snapshot/train_iter_'+str(i))] + '/'):
		os.makedirs('pred_visual/iter_'+model_file[len('snapshot/train_iter_'):len('snapshot/train_iter_'+str(i))] + '/')
	model_acc[idx] = test_accuracy(model_file)
	#
	plt.clf()
	plt.plot(model_acc[:(idx+1)])
	plt.ylabel('accuracy')
	plt.title('accuracy on '+dataset)
	plt.savefig('test_accuracy.png')
	
