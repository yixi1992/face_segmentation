import numpy as np
from PIL import Image
import glob
import scipy
import os
import random
import lmdb


import sys
sys.path.append("/lustre/yixi/face_segmentation_finetune/utils/")
from eval_utils import *


if __name__=='__main__':

	resize = True
	RSize = (200, 200)
	LabelSize = (200, 200)
	nopadding = False
	use_flow = ['f1']
	flow_dirs = ['flow_x', 'flow_y']
	RGB_mean_pad = False
	flow_mean_pad = True
	# Default is RGB_mean_pad = False and flow_mean_pad = True
	
	RGB_pad_values = [] if RGB_mean_pad else [0,0,0]
	flow_pad_value = 128 if flow_mean_pad else 0



	work_dir = '.'
	lmdb_dir = os.path.join(work_dir, '../camvidfmp200200epicflow_lmdb')
	deploy_file = os.path.join(work_dir, 'deploy.prototxt')
	snapshot = os.path.join(work_dir, 'snapshots_camvidfmp200200epicflow/modeldefaultflowsurg_lr1e-10/_iter_{snapshot_id}.caffemodel')
	iter = range(38000, 37000, -1000)
	
	
	model = snapshot.replace(work_dir, '').replace('snapshots_','').replace('/','_').replace('/_iter_{snapshot_id}.caffemodel','')
	pred_visual_dir_template = snapshot.replace('snapshots','pred_visual').replace('.caffemodel','')
		
	train_data = '/lustre/yixi/data/CamVid/701_StillsRaw_full/{id}.png'
	train_label_data = '/lustre/yixi/data/CamVid/label/indexedlabel/{id}_L.png'
	flow_data = '/lustre/yixi/data/CamVid/flow_all/flow/{id}.{flow_type}.{flow_dir}.png'
	
	
	
	
	args = CArgs()
	args.use_gpu = False
	args.resize = resize
	args.RSize = RSize
	args.LabelSize = LabelSize
	args.nopadding = nopadding
	args.use_flow = use_flow
	args.RGB_mean_pad = RGB_mean_pad
	args.flow_mean_pad =flow_mean_pad
	args.RGB_pad_values = RGB_pad_values
	args.flow_pad_value = flow_pad_value	
	args.BoxSize = (960, 960)  # None is padding to the square of the longer edge
	args.NumLabels = 32  # [0,31]
	args.BackGroundLabel = 32
	args.numclasses = 33

	#args.interested_class = range(0, args.NumLabels)
	args.interested_class = [2, 4, 5, 8, 9, 16, 17, 19, 20, 21, 26]
	args.eval_metric = 'eval_miu'
	args.RGB_mean_values = [78.1049806452, 76.5400646313, 74.0029471198]
	args.shortcut_inference = True
	args.work_dir = work_dir
	args.deploy_file = deploy_file
	args.model = model
	args.snapshot = snapshot
	args.pred_visual_dir_template = pred_visual_dir_template
	args.iter = iter
	args.test_ratio = 1.0 #will test #test size *0.1 samples



	inputs_all = [(os.path.splitext(os.path.basename(x))[0], x) for x in sorted(glob.glob( train_data.format(id='*')))]
	
	inputs_Train_keys = LMDB2Dict(os.path.join(lmdb_dir,'train-lmdb')).keys()
	
	inputs_Test = dict([(i, y) for (i,y) in inputs_all if not (i in inputs_Train_keys)])
	inputs_Train = dict([(i, y) for (i,y) in inputs_all if (i in inputs_Train_keys)])
	
	inputs_Train_Label = dict([(id, train_label_data.format(id=id)) for id in inputs_Train])
	inputs_Test_Label = dict([(id, train_label_data.format(id=id)) for id in inputs_Test])

	inputs_Train_flows =[dict([(id, flow_data.format(id=id, flow_type=flow_type, flow_dir=flow_dir)) for id in inputs_Train]) for flow_dir in flow_dirs for flow_type in use_flow ]
	inputs_Test_flows =[dict([(id, flow_data.format(id=id, flow_type=flow_type, flow_dir=flow_dir)) for id in inputs_Test]) for flow_dir in flow_dirs for flow_type in use_flow ]



	args.dataset_name = 'Test'
	eval(inputs_Test, inputs_Test_flows, inputs_Test_Label, args)
	inputs_Test.clear()
	inputs_Test_Label.clear()
	inputs_Test_flows.clear()

	args.dataset_name = 'Train'
	#eval(inputs_Train, inputs_Train_flows, inputs_Train_Label, args)
	inputs_Train.clear()
	inputs_Train_Label.clear()
	inputs_Train_flows.clear()

