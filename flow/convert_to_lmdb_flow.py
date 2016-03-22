#https://gist.github.com/longjon/ac410cad48a088710872#file-readme-md
 
#convert_to_lmdb.py
import caffe
import lmdb
from PIL import Image
import numpy as np
import glob
from random import shuffle
import scipy.io as sio
import scipy
import sys
import os
import shutil
import random

resize = True
# NumberTrain = 20 # Number of Training Images
NumberTest = 50 # Number of Testing Images
RSize = (300, 300)
LabelSize = (300, 300)
NumLabels = 32

if False:
	lmdb_dir = 'mass_lmdb'
	train_data = '/lustre/yixi/data/massimomauro-FASSEG-dataset-f93e332/V2/Train_RGB/{id}.bmp'
	test_data = '/lustre/yixi/data/massimomauro-FASSEG-dataset-f93e332/V2/Test_RGB/{id}.bmp'
	train_label_data = '/lustre/yixi/data/massimomauro-FASSEG-dataset-f93e332/V2/Train_Labels/labels/{id}.png'
	test_label_data = '/lustre/yixi/data/massimomauro-FASSEG-dataset-f93e332/V2/Test_Labels/labels/{id}.png'
	
	inputs_Train = [(os.path.splitext(os.path.basename(x))[0], x) for x in sorted(glob.glob( train_data.format(id='*')))]
	inputs_Test = [(os.path.splitext(os.path.basename(x))[0], x) for x in sorted(glob.glob( test_data.format(id='*')))]
	inputs_Train_Label = [(id, train_label_data.format(id=id)) for (id,y) in inputs_Train]
	inputs_Test_Label = [(id, test_label_data.format(id=id)) for (id,y) in inputs_Test]


if True:
	lmdb_dir = 'camvid300flow_lmdb'
	train_data = '/lustre/yixi/data/CamVid/701_StillsRaw_full/{id}.png'
	train_label_data = '/lustre/yixi/data/CamVid/label/indexedlabel/{id}_L.png'
	flow_x = '/lustre/yixi/data/CamVid/flow/{id}.flow_x.png'
	flow_y = '/lustre/yixi/data/CamVid/flow/{id}.flow_y.png'
	
	BoxSize = (960, 960)
	
	
	inputs_Train = [(os.path.splitext(os.path.basename(x))[0], x) for x in sorted(glob.glob( train_data.format(id='*')))]
	shuffle(inputs_Train)
	inputs_Test = inputs_Train[:NumberTest]
	inputs_Train = inputs_Train[NumberTest:]
	inputs_Train_Label = [(id, train_label_data.format(id=id)) for (id,y) in inputs_Train]
	inputs_Test_Label = [(id, train_label_data.format(id=id)) for (id,y) in inputs_Test]
	
	flow_x_Train = dict([(id, flow_x.format(id=id)) for (id,y) in inputs_Train])
	flow_x_Test = dict([(id, flow_x.format(id=id)) for (id,y) in inputs_Test])
	flow_y_Train = dict([(id, flow_y.format(id=id)) for (id,y) in inputs_Train])
	flow_y_Test = dict([(id, flow_y.format(id=id)) for (id,y) in inputs_Test])
	

class Resizer:
	def __init__(self, size, box_size):
		self.size = size
		self.box_size = box_size
		
	def padarray(self, im):
		pad_size = (self.box_size[0]-im.shape[0], self.box_size[1]-im.shape[1])
		if im.ndim==2:
			pad_im = np.pad(im, pad_width = ((0, pad_size[0]), (0, pad_size[1])), mode='constant', constant_values = NumLabels)
		else:
			pad_im = np.pad(im, pad_width = ((0, pad_size[0]), (0, pad_size[1]), (0,0)), mode='constant', constant_values = NumLabels)
		return pad_im

class ImageResizer(Resizer):
	def resize(self, im):
		pad_im = self.padarray(im)
		pad_im = Image.fromarray(pad_im)
		res_im = pad_im.resize(self.size, Image.ANTIALIAS)
		return np.array(res_im)

class LabelResizer(Resizer):
	def resize(self, im):
		pad_im = self.padarray(im)
		pad_im = Image.fromarray(pad_im)
		res_im = pad_im.resize(self.size, Image.NEAREST)
		return np.array(res_im)




def createLMDB(in_db, inputs_Train, flow_x={}, flow_y={}, resize=False, isLabel=False):
	RGB_sum = np.zeros(5)
	with in_db.begin(write=True) as in_txn:
		for (in_idx, in_) in inputs_Train:
			im = np.array(Image.open(in_))
#			print in_idx, in_
			if isLabel:
				Dtype = im.dtype
				
				if resize:
					res = LabelResizer(LabelSize, BoxSize)
					im = res.resize(im)
				
				im = im.reshape(im.shape[0],im.shape[1],1)
				im = np.array(im, Dtype)
				print np.amin(im),np.amax(im), im.shape
			else:
				Dtype = im.dtype
				im = im[:,:,::-1]	# reverse channels of image data
				
				# add flow TODO
				if (flow_x!={}):
					flow_im = np.array(Image.open(flow_x[in_idx]))
					flow_im = np.reshape(flow_im, (flow_im.shape[0], flow_im.shape[1], 1))
					im = np.concatenate((im, flow_im), axis=2)
					
				if (flow_y!={}):
					flow_im = np.array(Image.open(flow_y[in_idx]))
					flow_im = np.reshape(flow_im, (flow_im.shape[0], flow_im.shape[1], 1))
					im = np.concatenate((im, flow_im), axis=2)
				
				if resize:
					res = ImageResizer(RSize, BoxSize)
					im_res = res.resize(im[:,:,:3])
					for i in range(3, im.shape[2]):
						flow_im_res = res.resize(im[:,:,i])
						flow_im_res = np.reshape(flow_im_res, (flow_im_res.shape[0], flow_im_res.shape[1], 1))
						im_res = np.concatenate((im_res, flow_im_res), axis=2)
					im = im_res
				
				im = np.array(im,Dtype)
				#print im.shape
				RGB_sum = RGB_sum + np.mean(im, axis=(0,1))
			
			im = im.transpose((2,0,1))
			im_dat = caffe.io.array_to_datum(im)
			in_txn.put(in_idx,im_dat.SerializeToString())
	in_db.close()
	print 'mean RGB=',RGB_sum/len(inputs_Train)



if os.path.exists(lmdb_dir):
	shutil.rmtree(lmdb_dir, ignore_errors=True)

os.makedirs(lmdb_dir)

############################# Creating LMDB for Training Data ##############################

print("Creating Training Data LMDB File ..... ")

in_db = lmdb.open(os.path.join(lmdb_dir,'train-lmdb'), map_size=int(1e14))
createLMDB(in_db, inputs_Train, flow_x=flow_x_Train, flow_y=flow_y_Train, resize=resize)

 
############################# Creating LMDB for Training Labels ##############################
 
print("Creating Training Label LMDB File ..... ")
 
in_db = lmdb.open(os.path.join(lmdb_dir,'train-label-lmdb'), map_size=int(1e12))
createLMDB(in_db, inputs_Train_Label, resize=resize, isLabel=True)


############################# Creating LMDB for Testing Data ##############################

print("Creating Testing Data LMDB File ..... ")

in_db = lmdb.open(os.path.join(lmdb_dir,'test-lmdb'), map_size=int(1e14))
createLMDB(in_db, inputs_Test, flow_x=flow_x_Test, flow_y=flow_y_Test, resize=resize)


############################# Creating LMDB for Testing Labels ##############################

print("Creating Testing Label LMDB File ..... ")

in_db = lmdb.open(os.path.join(lmdb_dir,'test-label-lmdb'), map_size=int(1e12))
createLMDB(in_db, inputs_Test_Label, resize=resize, isLabel=True)
