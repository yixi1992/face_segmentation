# Make sure that caffe is on the python path:
caffe_root = './caffe'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe


# Load the fully convolutional network to transplant the parameters.
net_full_conv = caffe.Net('deploy_modeldefault.prototxt', 
                          '/lustre/yixi/face_segmentation_finetune/gcfv_flow/modeldefault/snapshots_gcfvshuffle200200/vgg_lr1e-10/_iter_30000.caffemodel', caffe.TEST)
net = caffe.Net('/lustre/yixi/face_segmentation_finetune/gcfv_flow/modeldefault/deploy_modeldefault.prototxt', '/lustre/yixi/face_segmentation_finetune/gcfv_flow/modeldefault/snapshots_gcfvshuffle200200/vgg_lr1e-10/_iter_30000.caffemodel', caffe.TEST)


params_full_conv = ['conv1_1_flow']
conv_params = {pr: [net_full_conv.params[pr][i].data for i in range(len(net_full_conv.params[pr]))] for pr in params_full_conv}
for conv in params_full_conv:
	print '{} weights are {} dimensional'.format(conv, conv_params[conv][0].shape) + ('' if len(conv_params[conv])<=1 else 'and biases are {} dimensional'.format(conv_params[conv][1].shape))


# Load the original network and extract the fully connected layers' parameters.
params = ['conv1_1']
fc_params = {pr: [net.params[pr][i].data for i in range(len(net.params[pr]))] for pr in params}
for fc in params:
	print '{} weights are {} dimensional'.format(fc, fc_params[fc][0].shape) + ('' if len(fc_params[fc])<=1 else 'and biases are {} dimensional'.format(fc_params[fc][1].shape))




for pr, pr_conv in zip(params, params_full_conv):
	for i in range(len(conv_params[pr_conv])):
		if len(fc_params[pr][i].shape)==4:
			conv_params[pr_conv][i][0:fc_params[pr][i].shape[0], 0:fc_params[pr][i].shape[1], 0:fc_params[pr][i].shape[2], 0:fc_params[pr][i].shape[3]] = fc_params[pr][i]
		elif len(fc_params[pr][i].shape)==1:
			conv_params[pr_conv][i][0:fc_params[pr][i].shape[0]] = fc_params[pr][i]
		else:
			print 'no------------------------ the shape is ', fc_params[pr][i].shape



net_full_conv.save('vgg_modeldefault_convflow_xavier_surg.caffemodel')

