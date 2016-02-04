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



blob = caffe.proto.caffe_pb2.BlobProto()
data=open('/lustre/yixi/origin_mean.binaryproto','rb').read()
blob.ParseFromString(data)
mean_arr = np.array( caffe.io.blobproto_to_array(blob) )


NumberTrain = 20 # Number of Training Images
NumberTest = 50 # Number of Testing Images
Rheight = 100 # Required Height
Rwidth = 100 # Required Width
RheightLabel = 100 # Height for the label
RwidthLabel = 100 # Width for the label
LabelWidth = 100 # Downscaled width of the label
LabelHeight = 100 # Downscaled height of the label

inputs_data_train = sorted(glob.glob('/lustre/yixi/data/massimomauro-FASSEG-dataset-f93e332/V2/Train_RGB/*.bmp'))
inputs_data_valid =  sorted(glob.glob('/lustre/yixi/data/massimomauro-FASSEG-dataset-f93e332/V2/Test_RGB/*.bmp'))


#shuffle(inputs_data_train) # Shuffle the DataSet
#shuffle(inputs_data_valid) # Shuffle the DataSet

#inputs_Train = inputs_data_train # Extract the training data from the complete set
#inputs_Test = inputs_data_valid # Extract the testing data from the complete set

inputs_Train = inputs_data_train[:NumberTrain] # Extract the training data from the complete set
inputs_Test = inputs_data_valid[:NumberTest] # Extract the testing data from the complete set
print len(inputs_Train)


############################# Creating LMDB for Training Data ##############################

print("Creating Training Data LMDB File ..... ")

in_db = lmdb.open('lmdb_data/train-lmdb',map_size=int(1e14))

with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs_Train):
        im = np.array(Image.open(in_)) # or load whatever ndarray you need
        print in_idx, in_
        Dtype = im.dtype
        im = im[:,:,::-1]
        im = Image.fromarray(im)
        im = im.resize([Rheight, Rwidth], Image.ANTIALIAS)
        im = np.array(im,Dtype)     
        im = im.transpose((2,0,1))
	im = im-mean_arr[0]
        im_dat = caffe.io.array_to_datum(im)
        in_txn.put('{:0>10d}'.format(in_idx),im_dat.SerializeToString())

in_db.close()

 
############################# Creating LMDB for Training Labels ##############################
 
print("Creating Training Label LMDB File ..... ")
 
in_db = lmdb.open('lmdb_data/train-label-lmdb',map_size=int(1e14))
 
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs_Train):
        in_label = '/lustre/yixi/data/massimomauro-FASSEG-dataset-f93e332/V2/Train_Labels/labels/'+in_[len('/lustre/yixi/data/massimomauro-FASSEG-dataset-f93e332/V2/Train_RGB/'):(len(in_)-len('.bmp'))]+'.png'
        print in_, in_label
        L = np.array(Image.open(in_label)) # or load whatever ndarray you need
        Dtype = L.dtype
        Limg = Image.fromarray(L)
        Limg = Limg.resize([LabelHeight, LabelWidth],Image.NEAREST) # To resize the Label file to the required size 
        L = np.array(Limg,Dtype)
        L = L.reshape(L.shape[0],L.shape[1],1)
        L = L.transpose((2,0,1))
        print ' max value=', np.amax(L), ' min value=', np.amin(L)
        L_dat = caffe.io.array_to_datum(L)
        in_txn.put('{:0>10d}'.format(in_idx),L_dat.SerializeToString())

in_db.close()

############################# Creating LMDB for Testing Data ##############################

print("Creating Testing Data LMDB File ..... ")

in_db = lmdb.open('lmdb_data/test-lmdb',map_size=int(1e12))

with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs_Test):
        im = np.array(Image.open(in_)) # or load whatever ndarray you need
        print in_idx, in_
        Dtype = im.dtype
        im = im[:,:,::-1]
        im = Image.fromarray(im)
        im = im.resize([Rheight, Rwidth], Image.ANTIALIAS)
        im = np.array(im,Dtype)     
        im = im.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(im)
	im = im-mean_arr[0]
        in_txn.put('{:0>10d}'.format(in_idx),im_dat.SerializeToString())

in_db.close()


############################# Creating LMDB for Testing Labels ##############################

print("Creating Testing Label LMDB File ..... ")

in_db = lmdb.open('lmdb_data/test-label-lmdb',map_size=int(1e14))

with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs_Test):
        in_label = '/lustre/yixi/data/massimomauro-FASSEG-dataset-f93e332/V2/Test_Labels/labels/'+in_[len('/lustre/yixi/data/massimomauro-FASSEG-dataset-f93e332/V2/Test_RGB/'):(len(in_)-len('.bmp'))]+'.png'
        print in_, in_label
        L = np.array(Image.open(in_label)) # or load whatever ndarray you need
        Dtype = L.dtype
        Limg = Image.fromarray(L)
        Limg = Limg.resize([LabelHeight, LabelWidth],Image.NEAREST) # To resize the Label file to the required size 
        L = np.array(Limg,Dtype)
        L = L.reshape(L.shape[0],L.shape[1],1)
        L = L.transpose((2,0,1))
        print ' max value=', np.amax(L), ' min value=', np.amin(L)
        L_dat = caffe.io.array_to_datum(L)
        in_txn.put('{:0>10d}'.format(in_idx),L_dat.SerializeToString())

in_db.close()
