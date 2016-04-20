import os, glob
import numpy as np
from PIL import Image
from collections import namedtuple
from random import shuffle
import shutil


import sys
sys.path.append('/lustre/yixi/face_segmentation_finetune/utils')
from convert_to_lmdb import *


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]


def Label2trainId(label_data):
	# map 225 to 19!
	id2trainId   = {label.id: label.trainId if label.trainId!=255 else 19 for label in labels}
	inputs_Label = dict([(os.path.splitext(os.path.basename(x))[0].replace('_gtFine_labelIds',''), x) for x in sorted(glob.glob(label_data.format(vid='*', fid='*')))])
	count = 0
	for k in inputs_Label:
		print k+' in '+len(inputs_Label)
		count += 1
		if count%proc_size!=proc_rank:
			continue
		label_path = inputs_Label[k]
		save_path = label_path.replace('_gtFine_labelIds','_gtFine_trainIds')
		if os.path.exists(save_path):
			continue
		im = np.array(Image.open(label_path), dtype=np.uint8)
		im = [map(lambda (x): id2trainId[x] , x) for x in im]
		#print im
		im = Image.fromarray(np.array(im, dtype=np.uint8))
		print k, ': save to '+ save_path
		im.save(save_path)	

if __name__=='__main__':
	if False:
		proc_rank = int(sys.argv[1])
		proc_size = int(sys.argv[2])
		Label2trainId('/lustre/yixi/data/cityscape/gtFine/train/{vid}/{vid}_{fid}_gtFine_labelIds.png')
		Label2trainId('/lustre/yixi/data/cityscape/gtFine/val/{vid}/{vid}_{fid}_gtFine_labelIds.png')
		Label2trainId('/lustre/yixi/data/cityscape/gtFine/test/{vid}/{vid}_{fid}_gtFine_labelIds.png')
		

	if False:
		print 'convert to lmdb begins....\n'
		resize = True
		RSize = (200, 200)
		LabelSize = (200, 200)
		nopadding = False
		use_flow = []
		flow_dirs = ['flow_x', 'flow_y']
		RGB_mean_pad = False
		flow_mean_pad = True
		# Default is RGB_mean_pad = False and flow_mean_pad = True
		
		RGB_pad_values = [] if RGB_mean_pad else [0,0,0]
		flow_pad_value = 128 if flow_mean_pad else 0


		lmdb_dir = 'cityscape' + ('rgbmp' if RGB_mean_pad else '') + ('fmp' if flow_mean_pad else '') + str(RSize[0]) + str(RSize[1]) + (''.join(use_flow)) + ('np' if nopadding else '') + '_lmdb'
			
		args = CArgs()
		args.resize = resize
		args.RSize = RSize
		args.LabelSize = LabelSize
		args.nopadding = nopadding
		args.useflow = useflow
		args.RGB_mean_pad = RGB_mean_pad
		args.flow_mean_pad = flow_mean_pad
		args.RGB_pad_values = RGB_pad_values
		args.flow_pad_value = flow_pad_value
		args.BoxSize = None # None is padding to the square of the longer edge
		args.NumLabels = 19 # [0,19]
		args.BackGroundLabel = 20
		args.lmdb_dir = lmdb_dir
		#args.proc_rank = proc_rank
		#args.proc_size = proc_size		


		train_data = '/lustre/yixi/data/cityscape/leftImg8bit/train/{vid}/{vid}_{fid}_leftImg8bit.png'
		train_label_data = '/lustre/yixi/data/cityscape/gtFine/train/{vid}/{vid}_{fid}_gtFine_trainIds.png'
		val_data = '/lustre/yixi/data/cityscape/leftImg8bit/val/{vid}/{vid}_{fid}_leftImg8bit.png'
		val_label_data = '/lustre/yixi/data/cityscape/gtFine/val/{vid}/{vid}_{fid}_gtFine_trainIds.png'
		test_data = '/lustre/yixi/data/cityscape/leftImg8bit/test/{vid}/{vid}_{fid}_leftImg8bit.png'
		test_label_data = '/lustre/yixi/data/cityscape/gtFine/test/{vid}/{vid}_{fid}_gtFine_trainIds.png'
		
		flow_data = 'TODO'

	
		inputs_Train = dict([(os.path.splitext(os.path.basename(x))[0].replace('_leftImg8bit',''), x) for x in sorted(glob.glob( train_data.format(vid='*',fid='*')))])
		inputs_Val = dict([(os.path.splitext(os.path.basename(x))[0].replace('_leftImg8bit',''), x) for x in sorted(glob.glob( val_data.format(vid='*',fid='*')))])
		inputs_Test = dict([(os.path.splitext(os.path.basename(x))[0].replace('_leftImg8bit',''), x) for x in sorted(glob.glob( test_data.format(vid='*',fid='*')))])
		

		inputs_Train_Label = dict([(os.path.splitext(os.path.basename(x))[0].replace('_gtFine_trainIds',''), x) for x in sorted(glob.glob( train_label_data.format(vid='*', fid='*')))])
		inputs_Val_Label = dict([(os.path.splitext(os.path.basename(x))[0].replace('_gtFine_trainIds',''), x) for x in sorted(glob.glob( val_label_data.format(vid='*', fid='*')))])
		inputs_Test_Label = dict([(os.path.splitext(os.path.basename(x))[0].replace('_gtFine_trainIds',''), x) for x in sorted(glob.glob( test_label_data.format(vid='*', fid='*')))])
		
		
		Train_keys = [i for i in inputs_Train.keys() if i in inputs_Train_Label.keys()]
		shuffle(Train_keys, lambda: 0.5)
		Val_keys = [i for i in inputs_Val.keys() if i in inputs_Val_Label.keys()]
		shuffle(Val_keys, lambda: 0.1)
		Test_keys = [i for i in inputs_Test.keys() if i in inputs_Test_Label.keys()]
		shuffle(Test_keys, lambda: 0.8)
	
		
		flow_Train = [dict([(id, flow_data.format(id=id, flow_type=flow_type, flow_dir=flow_dir)) for id in Train_keys]) for flow_dir in flow_dirs for flow_type in use_flow] 
		flow_Val = [dict([(id, flow_data.format(id=id, flow_type=flow_type, flow_dir=flow_dir)) for id in Val_keys]) for flow_dir in flow_dirs for flow_type in use_flow] 
		flow_Test = [dict([(id, flow_data.format(id=id, flow_type=flow_type, flow_dir=flow_dir)) for id in Test_keys]) for flow_dir in flow_dirs for flow_type in use_flow] 	



		if os.path.exists(lmdb_dir):
			shutil.rmtree(lmdb_dir, ignore_errors=True)

		if not os.path.exists(lmdb_dir):
			os.makedirs(lmdb_dir)

		############################ Creating LMDB for Training Data ##############################
		print("Creating Training Data LMDB File ..... ")
		createLMDBImage(os.path.join(lmdb_dir,'train-lmdb'), int(1e13), inputs_Train, flows=flow_Train,  keys=Train_keys, args=args)

		 
		############################# Creating LMDB for Training Labels ##############################
		print("Creating Training Label LMDB File ..... ")
		createLMDBLabel(os.path.join(lmdb_dir,'train-label-lmdb'), int(1e12), inputs_Train_Label, keys=Train_keys, args=args)


		############################# Creating LMDB for Validation Data ##############################
		print("Creating Validation Data LMDB File ..... ")
		createLMDBImage(os.path.join(lmdb_dir,'val-lmdb'), int(1e13), inputs_Val, flows=flow_Val,  keys=Val_keys, args=args)

		 
		############################# Creating LMDB for Validation Labels ##############################
		print("Creating Validation Label LMDB File ..... ")
		createLMDBLabel(os.path.join(lmdb_dir,'val-label-lmdb'), int(1e12), inputs_Val_Label, keys=Val_keys, args=args)

		
		############################# Creating LMDB for Testing Data ##############################
		print("Creating Testing Data LMDB File ..... ")
		createLMDBImage(os.path.join(lmdb_dir,'test-lmdb'), int(1e13), inputs_Test, flows=flow_Test, keys=Test_keys, args=args)


		############################# Creating LMDB for Testing Labels ##############################
		print("Creating Testing Label LMDB File ..... ")
		createLMDBLabel(os.path.join(lmdb_dir,'test-label-lmdb'), int(1e12), inputs_Test_Label, keys=Test_keys, args=args)
