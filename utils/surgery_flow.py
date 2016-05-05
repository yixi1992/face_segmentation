import caffe
import sys, getopt


#fromdeploy_file = '/scratch/groups/lsdavis/yixi/face_ss/models/gcfv/modelrgb/deploy_modeldefault.prototxt'
#caffemodel_file = '/scratch/groups/lsdavis/yixi/face_ss/models/gcfv/modelrgb/snapshots_gcfv500500/vgg_lr1e-10/_iter_8000.caffemodel'
#todeploy_file = 'deploy_modeldefault.prototxt'

def surgery(fromdeploy_file, todeploy_file, caffemodel_file, output_file, fl, tl):

	# Load the fully convolutional network to transplant the parameters.
	net_full_conv = caffe.Net(todeploy_file, caffemodel_file, caffe.TEST)
	net = caffe.Net(fromdeploy_file, caffemodel_file, caffe.TEST)


	params_full_conv = tl
	conv_params = {pr: [net_full_conv.params[pr][i].data for i in range(len(net_full_conv.params[pr]))] for pr in params_full_conv}
	for conv in params_full_conv:
		print '{} weights are {} dimensional'.format(conv, conv_params[conv][0].shape) + ('' if len(conv_params[conv])<=1 else 'and biases are {} dimensional'.format(conv_params[conv][1].shape))


	# Load the original network and extract the fully connected layers' parameters.
	params = fl
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

	net_full_conv.save(output_file)


def usage():
	print 'surgery_flow.py -f <fromdeploy> -t <todeploy> -c <caffemodel> -o <output> --fromlayer=<fromlayer> --tolayer=<tolayer>'
      	sys.exit(2)
   	
def main(argv):
	fr = ''
   	to = ''
	cm = ''
	o = ''
	fl = []
	tl = []
   	try:
      		opts, args = getopt.getopt(argv,'f:t:c:o:h',['fromdeploy=','todeploy=','caffemodel=','output=','fromlayer=','tolayer=','help'])
   	except getopt.GetoptError:
      		usage()
      	for opt, arg in opts:
		if opt in ('-h', '--help'):
			usage()
      		elif opt in ('-f', '--fromdeploy'):
			fr = arg
      		elif opt in ('-t', '--todeploy'):
         		to = arg
     		elif opt in ('-c', '--caffemodel'):
         		cm = arg
		elif opt in ('-o', '--output'):
			o = arg
		elif opt in ('--fromlayer'):
			fl = arg.split(',')
		elif opt in ('--tolayer'):
			tl = arg.split(',')
		else:
			print opt, arg
			usage()
	surgery(fr, to, cm, o, fl, tl)

if __name__=="__main__":
	main(sys.argv[1:])
