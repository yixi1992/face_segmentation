caffe_root = './caffe'
import sys
sys.path.insert(0, caffe_root+'/python/')
import caffe

net = caffe.Net('/lustre/yixi/face_segmentation_finetune/flow/modeldefault/deploy.prototxt', 
                '/lustre/yixi/face_segmentation_finetune/flow/modeldefault/snapshots_camvid200200/train_lr1e-10/_iter_77000.caffemodel', caffe.TEST)

net_full_conv = caffe.Net('deploy.prototxt', 
                          '/lustre/yixi/face_segmentation_finetune/flow/modeldefault/snapshots_camvid200200/train_lr1e-10/_iter_77000.caffemodel', caffe.TEST)




params_full_conv = ['score59-gcfv', 'upscore2-gcfv', 'score-pool4-gcfv', 'upsample-fused-16-gcfv', 'score-pool3-gcfv', 'upsample-gcfv']
conv_params = {pr: [net_full_conv.params[pr][i].data for i in range(len(net_full_conv.params[pr]))] for pr in params_full_conv}
for conv in params_full_conv:
	print '{} weights are {} dimensional'.format(conv, conv_params[conv][0].shape) + ('' if len(conv_params[conv])<=1 else 'and biases are {} dimensional'.format(conv_params[conv][1].shape))
	for i in range(2, len(conv_params[conv])):
		print 'additional weights are {}'.format(conv_params[conv][i].shape)



params = ['score59bg', 'upscore2bg', 'score-pool4bg', 'upsample-fused-16bg', 'score-pool3bg', 'upsamplebg']
fc_params = {pr: [net.params[pr][i].data for i in range(len(net.params[pr]))] for pr in params}
for fc in params:
	print '{} weights are {} dimensional'.format(fc, fc_params[fc][0].shape) + ('' if len(fc_params[fc])<=1 else 'and biases are {} dimensional'.format(fc_params[fc][1].shape))
	for i in range(2, len(fc_params[fc])):
		print 'additional weights are {}'.format(fc_params[fc][i].shape)





for pr, pr_conv in zip(params, params_full_conv):
	for i in range(len(conv_params[pr_conv])):
		mshape = min(fc_params[pr][i].shape, conv_params[pr_conv][i].shape)
		print mshape
		if len(mshape)==4:
			#print mshape[0], mshape[1], mshape[2], mshape[3]
			conv_params[pr_conv][i][0:mshape[0], 0:mshape[1], 0:mshape[2], 0:mshape[3]] = fc_params[pr][i][0:mshape[0], 0:mshape[1], 0:mshape[2], 0:mshape[3]]
			#conv_params[pr_conv][i][0:min(fc_params[pr][i].shape[0], conv_params[pr_conv][i].shape[0]), 0:min(fc_params[pr][i].shape[1], conv_params[pr_conv][i].shape[1]), 0:min(fc_params[pr][i].shape[2], conv_params[pr_conv][i].shape[2]), 0:min(fc_params[pr][i].shape[3], conv_params[pr_conv][i].shape[3])] = fc_params[pr][i]
		elif len(mshape)==1:
			conv_params[pr_conv][i][0:mshape[0]] = fc_params[pr][i][0:mshape[0]]



net_full_conv.save('camvid_modeldefault_surg.caffemodel')

