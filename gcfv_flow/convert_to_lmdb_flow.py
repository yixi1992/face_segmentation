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




class Resizer:
	def __init__(self, size, box_size):
		self.size = size
		self.box_size = box_size
		
	def padarray(self, im, padval):
		if nopadding:
			return im
		
		box_size = self.box_size
		if box_size==None:
			box_size = (max(im.shape[0], im.shape[1]), max(im.shape[0], im.shape[1]))
		
		pad_size = (box_size[0]-im.shape[0], box_size[1]-im.shape[1])
		if im.ndim==2:
			pad_im = np.pad(im, pad_width = ((0, pad_size[0]), (0, pad_size[1])), mode='constant', constant_values = padval)
		else:
			pad_im = np.pad(im, pad_width = ((0, pad_size[0]), (0, pad_size[1]), (0,0)), mode='constant', constant_values = padval)
		return pad_im

class ImageResizer(Resizer):
	def resize(self, im):
		pad_im = self.padarray(im, 0)
		pad_im = Image.fromarray(pad_im)
		res_im = pad_im.resize(self.size, Image.ANTIALIAS)
		return np.array(res_im)

class LabelResizer(Resizer):
	def resize(self, im):
		pad_im = self.padarray(im, BackGroundLabel)
		pad_im = Image.fromarray(pad_im)
		res_im = pad_im.resize(self.size, Image.NEAREST)
		return np.array(res_im)

def LoadLabel(filename, resizer=None):
	im = np.array(Image.open(filename))
	Dtype = im.dtype
	
	if resizer!=None
		im = resizer.resize(im)
	
	im = im.reshape(im.shape[0],im.shape[1],1)
	im = np.array(im, Dtype)
	#print np.amin(im),np.amax(im), im.shape
	
	im = im.transpose((2,0,1))
	return im

def LoadImage(filename, flow_x=None, flow_y=None, resize=None):
	im = np.array(Image.open(in_))
	Dtype = im.dtype
	im = im[:,:,::-1]	# reverse channels of image data
	
	if (flow_x!=None):
		flow_im = np.array(Image.open(flow_x))
		flow_im = np.reshape(flow_im, (flow_im.shape[0], flow_im.shape[1], 1))
		im = np.concatenate((im, flow_im), axis=2)
		
	if (flow_y!=None):
		flow_im = np.array(Image.open(flow_y))
		flow_im = np.reshape(flow_im, (flow_im.shape[0], flow_im.shape[1], 1))
		im = np.concatenate((im, flow_im), axis=2)
	
	if resizer!=None:
		im_res = res.resize(im[:,:,:3])
		for i in range(3, im.shape[2]):
			flow_im_res = res.resize(im[:,:,i])
			flow_im_res = np.reshape(flow_im_res, (flow_im_res.shape[0], flow_im_res.shape[1], 1))
			im_res = np.concatenate((im_res, flow_im_res), axis=2)
		im = im_res
		
	im = np.array(im,Dtype)
	im = im.transpose((2,0,1))
	# shape channels*width*height, e.g. 3*640*420
	return im

def createLMDBLabel(dir, mapsize, inputs_Train, flow_x=None, flow_y=None, resize=False, keys=None):
	in_db = lmdb.open(dir, map_size=mapsize)
	resizer = None if !resize else LabelResizer(LabelSize, BoxSize)
	with in_db.begin(write=True) as in_txn:
		for (in_idx, key) in enumerate(keys):
			in_ = inputs_Train[key]
			im = LoadLabel(in_, resizer)
			im_dat = caffe.io.array_to_datum(im)
			in_txn.put(str(in_idx),im_dat.SerializeToString())
	in_db.close()



def createLMDBImage(dir, mapsize, inputs_Train, flow_x=None, flow_y=None, resize=False, keys=None):
	in_db = lmdb.open(dir, map_size=mapsize)
	RGB_sum = np.zeros(3 + (flow_x!=None) + (flow_y!=None))
	resizer = None if !resize else ImageResizer(RSize, BoxSize)
	with in_db.begin(write=True) as in_txn:
		for (in_idx, key) in enumerate(keys):
			in_ = inputs_Train[key]
			im = LoadImage(in_, flow_x[key], flow_y[key], resizer)
			RGB_sum = RGB_sum + np.mean(im, axis=(1,2))
			im_dat = caffe.io.array_to_datum(im)
			in_txn.put(str(in_idx),im_dat.SerializeToString())
	in_db.close()
	f = open(os.path.join(dir, 'RGB_mean'), 'w')
	RGB_mean = RGB_sum/len(inputs_Train)
	for i in range(RGB_mean.size):
		f.write('mean_value: ' + str(RGB_mean[i]) + '\n')
	for i in range(RGB_mean.size):
		f.write(str(RGB_mean[i]) + ', ')
	f.close()



if __name__=='__main__':
	resize = True
	# NumberTrain = 20 # Number of Training Images
	NumberTest = 50 # Number of Testing Images
	RSize = (200, 200)
	LabelSize = (200, 200)
	nopadding = False
	useflow = False

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


	if False:
		lmdb_dir = 'camvid' + str(RSize[0]) + str(RSize[1]) + ('flow' if useflow else '') + ('np' if nopadding else '') + '_lmdb'
		train_data = '/lustre/yixi/data/CamVid/701_StillsRaw_full/{id}.png'
		train_label_data = '/lustre/yixi/data/CamVid/label/indexedlabel/{id}_L.png'
		flow_x = '/lustre/yixi/data/CamVid/flow/{id}.flow_x.png'
		flow_y = '/lustre/yixi/data/CamVid/flow/{id}.flow_y.png'
		
		BoxSize = (960, 960)
		NumLabels = 32
		BackGroundLabel = 32

		inputs_Train = [(os.path.splitext(os.path.basename(x))[0], x) for x in sorted(glob.glob( train_data.format(id='*')))]
		shuffle(inputs_Train)
		inputs_Test = inputs_Train[:NumberTest]
		inputs_Train = inputs_Train[NumberTest:]
		inputs_Train_Label = [(id, train_label_data.format(id=id)) for (id,y) in inputs_Train]
		inputs_Test_Label = [(id, train_label_data.format(id=id)) for (id,y) in inputs_Test]
		
		if useflow:
			flow_x_Train = dict([(id, flow_x.format(id=id)) for (id,y) in inputs_Train])
			flow_x_Test = dict([(id, flow_x.format(id=id)) for (id,y) in inputs_Test])
			flow_y_Train = dict([(id, flow_y.format(id=id)) for (id,y) in inputs_Train])
			flow_y_Test = dict([(id, flow_y.format(id=id)) for (id,y) in inputs_Test])
		else:
			flow_x_Train = None
			flow_x_Test = None
			flow_y_Train = None
			flow_y_Test = None

	if True:
		lmdb_dir = 'gcfvshuffle' + str(RSize[0]) + str(RSize[1]) + ('flow' if useflow else '') + ('np' if nopadding else '') + '_lmdb'
		train_data = '/lustre/yixi/data/gcfv_dataset/cross_validation/videos/frames/{id}.jpg'
		train_label_data = '/lustre/yixi/data/gcfv_dataset/cross_validation/ground_truth/labels/{id}_gt.png'
		test_data = '/lustre/yixi/data/gcfv_dataset/external_validation/videos/frames/{id}.jpg'
		test_label_data = '/lustre/yixi/data/gcfv_dataset/external_validation/ground_truth/labels/{id}_gt.png'

		BoxSize = None # None is padding to the square of the longer edge

		NumLabels = 8
		BackGroundLabel = 0
		
		inputs_Train = dict([(os.path.splitext(os.path.basename(x))[0], x) for x in sorted(glob.glob( train_data.format(id='*')))])
		inputs_Train_Label = dict([(os.path.splitext(os.path.basename(x))[0].replace('_gt',''), x) for x in sorted(glob.glob( train_label_data.format(id='*')))])
		Train_keys = [i for i in inputs_Train.keys() if i in inputs_Train_Label.keys()]
		shuffle(Train_keys)
		print Train_keys

		inputs_Test = dict([(os.path.splitext(os.path.basename(x))[0], x) for x in sorted(glob.glob( test_data.format(id='*')))])
		inputs_Test_Label = dict([(os.path.splitext(os.path.basename(x))[0].replace('_gt',''), x) for x in sorted(glob.glob( test_label_data.format(id='*')))])
		Test_keys = [i for i in inputs_Test.keys() if i in inputs_Test_Label.keys()]
		shuffle(Test_keys)


		NumberTest = len(inputs_Test)

		print len(inputs_Train), '=', len(inputs_Train_Label), len(inputs_Test), '=', len(inputs_Test_Label)



		if useflow:
			flow_x_Train = dict([(id, flow_x.format(id=id)) for (id,y) in inputs_Train])
			flow_x_Test = dict([(id, flow_x.format(id=id)) for (id,y) in inputs_Test])
			flow_y_Train = dict([(id, flow_y.format(id=id)) for (id,y) in inputs_Train])
			flow_y_Test = dict([(id, flow_y.format(id=id)) for (id,y) in inputs_Test])
		else:
			flow_x_Train = None
			flow_x_Test = None
			flow_y_Train = None
			flow_y_Test = None



	if os.path.exists(lmdb_dir):
		shutil.rmtree(lmdb_dir, ignore_errors=True)

	os.makedirs(lmdb_dir)

	############################# Creating LMDB for Training Data ##############################
	print("Creating Training Data LMDB File ..... ")
	createLMDBImage(os.path.join(lmdb_dir,'train-lmdb'), int(1e14), inputs_Train, flow_x=flow_x_Train, flow_y=flow_y_Train, resize=resize, keys=Train_keys)

	 
	############################# Creating LMDB for Training Labels ##############################
	print("Creating Training Label LMDB File ..... ")
	createLMDBLabel(os.path.join(lmdb_dir,'train-label-lmdb'), int(1e12), inputs_Train_Label, resize=resize, keys=Train_keys)


	############################# Creating LMDB for Testing Data ##############################
	print("Creating Testing Data LMDB File ..... ")
	createLMDBImage(os.path.join(lmdb_dir,'test-lmdb'), int(1e14), inputs_Test, flow_x=flow_x_Test, flow_y=flow_y_Test, resize=resize, keys=Test_keys)


	############################# Creating LMDB for Testing Labels ##############################
	print("Creating Testing Label LMDB File ..... ")
	createLMDBLabel(os.path.join(lmdb_dir,'test-label-lmdb'), int(1e12), inputs_Test_Label, resize=resize, keys=Test_keys)
