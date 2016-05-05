import caffe









import caffe
import sys, getopt

def surgery(fromdeploy_file, todeploy_file, caffemodel_file, output_file):
	# Load the original network and extract the fully connected layers' parameters.

	net = caffe.Net(fromdeploy_file, caffemodel_file, caffe.TEST)

	params = ['fc6', 'fc7']
	# fc_params = {name: (weights, biases)}
	fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

	for fc in params:
		print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)


	# Load the fully convolutional network to transplant the parameters.
	net_full_conv = caffe.Net(todeploy_file, caffemodel_file, caffe.TEST)
	params_full_conv = ['fc6-conv', 'fc7-conv']

	conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

	for conv in params_full_conv:
		print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)



	for pr, pr_conv in zip(params, params_full_conv):
	    conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
	    conv_params[pr_conv][1][...] = fc_params[pr][1]



	net_full_conv.save(output_file)


def main(argv):
	fr = ''
   	to = ''
	cm = ''
	o = ''
   	try:
      		opts, args = getopt.getopt(argv,'f:t:c:o:h',['fromdeploy=','todeploy=','caffemodel=', 'output=', 'help'])
	except getopt.GetoptError:
      		print 'surgery.py -f <fromdeploy> -t <todeploy> -c <caffemodel> -o <output>'
      		sys.exit(2)
   	for opt, arg in opts:
		if opt in ('-h', '--help'):
			print 'surgery.py -f <fromdeploy> -t <todeploy> -c <caffemodel> -o <output>'
      			sys.exit(2)
      		elif opt in ('-f', '--fromdeploy'):
			fr = arg
      		elif opt in ('-t', '--todeploy'):
         		to = arg
     		elif opt in ('-c', '--caffemodel'):
         		cm = arg
		elif opt in ('-o', '--output'):
			o = arg
		else:
			print 'surgery.py -f <fromdeploy> -t <todeploy> -c <caffemodel> -o <output>'
      			sys.exit(2)
	surgery(fr, to, cm, o)

if __name__=="__main__":
	main(sys.argv[1:])
