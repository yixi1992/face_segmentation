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

resize = True
NumberTrain = 20 # Number of Training Images
NumberTest = 50 # Number of Testing Images
Rheight = 100 # Required Height
Rwidth = 100 # Required Width
RheightLabel = 100 # Height for the label
RwidthLabel = 100 # Width for the label
LabelWidth = 100 # Downscaled width of the label
LabelHeight = 100 # Downscaled height of the label

lmdb_dir = 'mass_lmdb'
train_data = '/lustre/yixi/data/massimomauro-FASSEG-dataset-f93e332/V2/Train_RGB/{id}.bmp'
test_data = '/lustre/yixi/data/massimomauro-FASSEG-dataset-f93e332/V2/Test_RGB/{id}.bmp'
train_label_data = '/lustre/yixi/data/massimomauro-FASSEG-dataset-f93e332/V2/Train_Labels/labels/{id}.png'
test_label_data = '/lustre/yixi/data/massimomauro-FASSEG-dataset-f93e332/V2/Test_Labels/labels/{id}.png'

inputs_Train = [(int(os.path.splitext(os.path.basename(x))[0]), x) for x in sorted(glob.glob( train_data.format(id='*')))]
inputs_Test = [(int(os.path.splitext(os.path.basename(x))[0]), x) for x in sorted(glob.glob( test_data.format(id='*')))]

inputs_Train_Label = [(id, train_label_data.format(id=str(id))) for (id,y) in inputs_Train]
inputs_Test_Label = [(id, test_label_data.format(id=str(id))) for (id,y) in inputs_Test]


def createLMDB(in_db, inputs_Train, resize=False, isLabel=False):
	with in_db.begin(write=True) as in_txn:
		for (in_idx, in_) in inputs_Train:
			im = np.array(Image.open(in_))
			print in_idx, in_
			Dtype = im.dtype
			if not isLabel:
				im = im[:,:,::-1]	# reverse channels of image data
			im = Image.fromarray(im)
			if resize:
				if not isLabel:
					im = im.resize([Rheight, Rwidth], Image.ANTIALIAS)
				else:
					im = im.resize([LabelHeight, LabelWidth],Image.NEAREST)
			im = np.array(im,Dtype)     
			if isLabel:
				im = im.reshape(im.shape[0],im.shape[1],1)
			im = im.transpose((2,0,1))
			im_dat = caffe.io.array_to_datum(im)
			in_txn.put('{:0>10d}'.format(in_idx),im_dat.SerializeToString())
	in_db.close()
	
	
	
if not os.path.exists(lmdb_dir):
		os.makedirs(lmdb_dir)
else:
	shutil.rmtree(lmdb_dir, ignore_errors=True)
############################# Creating LMDB for Training Data ##############################

print("Creating Training Data LMDB File ..... ")

in_db = lmdb.open(os.path.join(lmdb_dir,'train-lmdb'), map_size=int(1e14))
createLMDB(in_db, inputs_Train, resize)

 
############################# Creating LMDB for Training Labels ##############################
 
print("Creating Training Label LMDB File ..... ")
 
in_db = lmdb.open(os.path.join(lmdb_dir,'train-label-lmdb'), map_size=int(1e14))
createLMDB(in_db, inputs_Train_Label, resize, isLabel=True)


############################# Creating LMDB for Testing Data ##############################

print("Creating Testing Data LMDB File ..... ")

in_db = lmdb.open(os.path.join(lmdb_dir,'test-lmdb'), map_size=int(1e14))
createLMDB(in_db, inputs_Test, resize)


############################# Creating LMDB for Testing Labels ##############################

print("Creating Testing Label LMDB File ..... ")

in_db = lmdb.open(os.path.join(lmdb_dir,'test-label-lmdb'), map_size=int(1e14))
createLMDB(in_db, inputs_Test_Label, resize, isLabel=True)