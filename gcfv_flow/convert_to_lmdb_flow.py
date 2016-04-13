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
	def __init__(self, size, box_size, nopadding=False, RGB_pad_value=None, flow_pad_value=128):
		self.size = size
		self.box_size = box_size
		self.nopadding = nopadding
		self.RGB_pad_value = RGB_pad_value
		self.flow_pad_value = flow_pad_value
		
	def padarray(self, im, padval):
		if self.nopadding:
			return im
		
		box_size = self.box_size
		box_size = (max(im.shape[0], im.shape[1]), max(im.shape[0], im.shape[1])) if box_size==None else box_size
		
		pad_size = (box_size[0]-im.shape[0], box_size[1]-im.shape[1])
		if im.ndim==2: #flow or label
			pad_im = np.pad(im, pad_width = ((0, pad_size[0]), (0, pad_size[1])), mode='constant', constant_values = padval)
		else: #RGB
			arrays = [np.pad(im[:,:,ch], pad_width = ((0, pad_size[0]), (0, pad_size[1])), mode='constant', constant_values = padval[ch]) for ch in range(im.shape[2])]
			pad_im = np.stack(arrays, axis=2)
		return pad_im

class ImageResizer(Resizer):
	def resize(self, im, padval=None):
		if im.ndim==2:
			padval = self.flow_pad_value
		else:
			padval = self.RGB_pad_value
		pad_im = self.padarray(im, padval)
		pad_im = Image.fromarray(pad_im)
		res_im = pad_im.resize(self.size, Image.ANTIALIAS)
		return np.array(res_im)

class LabelResizer(Resizer):
	def resize(self, im):
		pad_im = self.padarray(im, BackGroundLabel)
		pad_im = Image.fromarray(pad_im)
		res_im = pad_im.resize(self.size, Image.NEAREST)
		return np.array(res_im)
	
	def upsample(self, im, ImgSize):
		im = Image.fromarray(im)
		if self.nopadding:
			res_im = im.resize(ImgSize, Image.NEAREST)
			return np.array(res_im)

		box_size = self.box_size
		box_size = (max(ImgSize[0], ImgSize[1]), max(ImgSize[0], ImgSize[1])) if box_size==None else box_size
		res_im = im.resize(box_size, Image.NEAREST)
		res_im = np.array(res_im)
		res_im = res_im[0:ImgSize[0], 0:ImgSize[1]]
		return res_im



def LoadLabel(filename, resizer=None):
	im = np.array(Image.open(filename))
	Dtype = im.dtype
	
	if resizer!=None:
		im = resizer.resize(im)
	
	im = im.reshape(im.shape[0],im.shape[1],1)
	im = np.array(im, Dtype)
	#print np.amin(im),np.amax(im), im.shape
	
	im = im.transpose((2,0,1))
	return im

def LoadImage(filename, flow_x=None, flow_y=None, resizer=None):
	im = np.array(Image.open(filename))
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
		im_res = resizer.resize(im[:,:,:3])
		for i in range(3, im.shape[2]):
			flow_im_res = resizer.resize(im[:,:,i], 128)
			flow_im_res = np.reshape(flow_im_res, (flow_im_res.shape[0], flow_im_res.shape[1], 1))
			im_res = np.concatenate((im_res, flow_im_res), axis=2)
		im = im_res
		
	im = np.array(im,Dtype)
	im = im.transpose((2,0,1))
	# shape channels*width*height, e.g. 3*640*420
	return im

def createLMDBLabel(dir, mapsize, inputs_Train, flow_x=None, flow_y=None, resize=False, keys=None):
	in_db = lmdb.open(dir, map_size=mapsize)
	resizer = None if not resize else LabelResizer(LabelSize, BoxSize, nopadding)
	with in_db.begin(write=True) as in_txn:
		for (in_idx, key) in enumerate(keys):
			in_ = inputs_Train[key]
			im = LoadLabel(in_, resizer)
			im_dat = caffe.io.array_to_datum(im)
			in_txn.put(str(in_idx),im_dat.SerializeToString())
			print in_idx, 'label', im.shape
	in_db.close()



def createLMDBImage(dir, mapsize, inputs_Train, flow_x=None, flow_y=None, resize=False, keys=None):
	in_db = lmdb.open(dir, map_size=mapsize)
	RGB_sum = np.zeros(3 + (flow_x!=None) + (flow_y!=None))
	resizer = None if not resize else ImageResizer(RSize, BoxSize, nopadding, mean_values)
	with in_db.begin(write=True) as in_txn:
		for (in_idx, key) in enumerate(keys):
			print in_idx
			in_ = inputs_Train[key]
			im = LoadImage(in_, flow_x[key] if flow_x!=None and (key in flow_x) else None, flow_y[key] if flow_y!=None and (key in flow_y) else None, resizer)
			RGB_sum = RGB_sum + np.mean(im, axis=(1,2))
			im_dat = caffe.io.array_to_datum(im)
			in_txn.put(str(in_idx),im_dat.SerializeToString())
			print in_idx, im.shape, RGB_sum/(in_idx+1)
	in_db.close()
	f = open(os.path.join(dir, 'RGB_mean'), 'w')
	RGB_mean = RGB_sum/len(keys)
	for i in range(RGB_mean.size):
		f.write('mean_value: ' + str(RGB_mean[i]) + '\n')
	for i in range(RGB_mean.size):
		f.write(str(RGB_mean[i]) + ', ')
	f.close()



if __name__=='__main__':
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
		resize = True
		NumberTest = 50 # Number of Testing Images
		RSize = (200, 200)
		LabelSize = (200, 200)
		nopadding = False
		useflow = True
		
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
		inputs_Test = dict(inputs_Train[:NumberTest])
		inputs_Train = dict(inputs_Train[NumberTest:])
		inputs_Train_Label = dict([(id, train_label_data.format(id=id)) for id in inputs_Train.keys()])
		inputs_Test_Label = dict([(id, train_label_data.format(id=id)) for id in inputs_Test.keys()])
		Train_keys = inputs_Train.keys()
		Test_keys = inputs_Test.keys()

		flow_x_Train = None if not useflow else dict([(id, flow_x.format(id=id)) for id in inputs_Train])
		flow_x_Test = None if not useflow else dict([(id, flow_x.format(id=id)) for id in inputs_Test])
		flow_y_Train = None if not useflow else dict([(id, flow_y.format(id=id)) for id in inputs_Train])
		flow_y_Test = None if not useflow else dict([(id, flow_y.format(id=id)) for id in inputs_Test])

	if True:
		resize = True
		RSize = (200, 200)
		LabelSize = (200, 200)
		nopadding = False
		useflow = False
		mean_values = [121.364250092, 126.289872692, 124.244447077]

		lmdb_dir = 'test_gcfvmeanpad' + str(RSize[0]) + str(RSize[1]) + ('flow' if useflow else '') + ('np' if nopadding else '') + '_lmdb'
		train_data = '/lustre/yixi/data/gcfv_dataset/cross_validation/videos/frames/{id}.jpg'
		train_label_data = '/lustre/yixi/data/gcfv_dataset/cross_validation/ground_truth/labels/{id}_gt.png'
		test_data = '/lustre/yixi/data/gcfv_dataset/external_validation/videos/frames/{id}.jpg'
		test_label_data = '/lustre/yixi/data/gcfv_dataset/external_validation/ground_truth/labels/{id}_gt.png'
		train_flow_x = '/lustre/yixi/data/gcfv_dataset/cross_validation/videos/flow/{id}.flow_x.png'
		train_flow_y = '/lustre/yixi/data/gcfv_dataset/cross_validation/videos/flow/{id}.flow_y.png'
		test_flow_x = '/lustre/yixi/data/gcfv_dataset/external_validation/videos/flow/{id}.flow_x.png'
		test_flow_y = '/lustre/yixi/data/gcfv_dataset/external_validation/videos/flow/{id}.flow_y.png'

		BoxSize = None # None is padding to the square of the longer edge
		NumLabels = 8
		BackGroundLabel = 0
		
		inputs_Train = dict([(os.path.splitext(os.path.basename(x))[0], x) for x in sorted(glob.glob( train_data.format(id='*')))])
		inputs_Train_Label = dict([(os.path.splitext(os.path.basename(x))[0].replace('_gt',''), x) for x in sorted(glob.glob( train_label_data.format(id='*')))])
		Train_keys = [i for i in inputs_Train.keys() if i in inputs_Train_Label.keys()]
		shuffle(Train_keys)

		inputs_Test = dict([(os.path.splitext(os.path.basename(x))[0], x) for x in sorted(glob.glob( test_data.format(id='*')))])
		inputs_Test_Label = dict([(os.path.splitext(os.path.basename(x))[0].replace('_gt',''), x) for x in sorted(glob.glob( test_label_data.format(id='*')))])
		Test_keys = [i for i in inputs_Test.keys() if i in inputs_Test_Label.keys()]
		shuffle(Test_keys)

		flow_x_Train = None if not useflow else dict([(id, train_flow_x.format(id=id)) for id in inputs_Train.keys()])
		flow_x_Test = None if not useflow else dict([(id, test_flow_x.format(id=id)) for id in inputs_Test.keys()])
		flow_y_Train = None if not useflow else dict([(id, train_flow_y.format(id=id)) for id in inputs_Train.keys()])
		flow_y_Test = None if not useflow else dict([(id, test_flow_y.format(id=id)) for id in inputs_Test.keys()])


		NumberTest = len(inputs_Test)

		print len(inputs_Train), '=', len(inputs_Train_Label), len(inputs_Test), '=', len(inputs_Test_Label)



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
