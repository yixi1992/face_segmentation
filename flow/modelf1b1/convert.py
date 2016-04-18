import os, glob
import numpy as np
from PIL import Image
from collections import namedtuple
from random import shuffle
import shutil


import sys
sys.path.append('/lustre/yixi/face_segmentation_finetune//utils')
from convert_to_lmdb import *

if __name__=='__main__':
	if True:
		print 'convert to lmdb begins....\n'
		resize = True
		RSize = (500, 500)
		LabelSize = (500, 500)
		nopadding = False
		useflow = True
		RGB_mean_pad = False
		flow_mean_pad = True
		# Default is RGB_mean_pad = False and flow_mean_pad = True
		
		RGB_pad_values = [] if RGB_mean_pad else [0,0,0]
		flow_pad_value = 128 if flow_mean_pad else 0


		lmdb_dir = 'camvid' + ('rgbmp' if RGB_mean_pad else '') + ('fmp' if flow_mean_pad else '') + str(RSize[0]) + str(RSize[1]) + ('flow' if useflow else '') + ('np' if nopadding else '') + '_lmdb'
			
		args = CArgs()
		args.resize = resize
		args.RSize = RSize
		args.LabelSize = LabelSize
		args.nopadding = nopadding
		args.useflow = useflow
		args.RGB_mean_pad = RGB_mean_pad
		args.flow_mean_pad =flow_mean_pad
		args.RGB_pad_values = RGB_pad_values
		args.flow_pad_value = flow_pad_value
		args.BoxSize = None # None is padding to the square of the longer edge
		args.NumLabels = 32 # [0,31]
		args.BackGroundLabel = 32
		args.lmdb_dir = lmdb_dir
		#args.proc_rank = proc_rank
		#args.proc_size = proc_size		

		NumberTest = 50
		train_data = '/lustre/yixi/data/CamVid/701_StillsRaw_full/{id}.png'
	 	train_label_data = '/lustre/yixi/data/CamVid/label/indexedlabel/{id}_L.png'
	 	flow_x = '/lustre/yixi/data/CamVid/flow_all/flow/{id}.f1.flow_x.png'
		flow_y = '/lustre/yixi/data/CamVid/flow_all/flow/{id}.f1.flow_y.png'
		
	
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




		if os.path.exists(lmdb_dir):
			shutil.rmtree(lmdb_dir, ignore_errors=True)

		if not os.path.exists(lmdb_dir):
			os.makedirs(lmdb_dir)

		############################# Creating LMDB for Training Data ##############################
		print("Creating Training Data LMDB File ..... ")
		createLMDBImage(os.path.join(lmdb_dir,'train-lmdb'), int(1e13), inputs_Train, flow_x=flow_x_Train, flow_y=flow_y_Train, keys=Train_keys, args=args)

		 
		############################# Creating LMDB for Training Labels ##############################
		print("Creating Training Label LMDB File ..... ")
		createLMDBLabel(os.path.join(lmdb_dir,'train-label-lmdb'), int(1e12), inputs_Train_Label, keys=Train_keys, args=args)

		
		############################# Creating LMDB for Testing Data ##############################
		print("Creating Testing Data LMDB File ..... ")
		createLMDBImage(os.path.join(lmdb_dir,'test-lmdb'), int(1e13), inputs_Test, flow_x=flow_x_Test, flow_y=flow_y_Test, keys=Test_keys, args=args)


		############################# Creating LMDB for Testing Labels ##############################
		print("Creating Testing Label LMDB File ..... ")
		createLMDBLabel(os.path.join(lmdb_dir,'test-label-lmdb'), int(1e12), inputs_Test_Label, keys=Test_keys, args=args)

