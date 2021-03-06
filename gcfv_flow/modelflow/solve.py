from __future__ import division
import caffe
import numpy as np

# make a bilinear interpolation kernel
# credit @longjon
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k:
            print 'input + output channels need to be the same'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt

# base net -- follow the editing model parameters example to make
# a fully convolutional VGG16 net.
# http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/net_surgery.ipynb
# from gcfv RGB trained /lustre/yixi/face_segmentation_finetune/gcfv_flow/modeldefault/snapshots_gcfvshuffle200200/train_lr1e-10/_iter_30000.caffemodel

#Camvid finetuned gcfv RGB
#base_weights = 'modeldefaultflow.caffemodel'

#vggRGB finetuned
#base_weights = 'vgg_modeldefault_surg.caffemodel' 
#base_weights = '/lustre/yixi/face_segmentation_finetune/gcfv_flow/modelflow/snapshots_gcfvshuffle200200/vgg_lr1e-14/_iter_5000.caffemodel'
#base_weights = '/lustre/yixi/face_segmentation_finetune/gcfv_flow/modelflow/snapshots_gcfvshuffle200200flow/vgg_lr1e-14/_iter_15000.caffemodel'

#base_weights = 'vgg_modeldefault_convflow_xavier_surg.caffemodel' #xavier initialize conv1_1_flow
#base_weights = '/lustre/yixi/face_segmentation_finetune/gcfv_flow/modelflow/snapshots_gcfvshuffle200200flow/vgg_convflow_xavier_lr1e-12/_iter_6000.caffemodel'
#base_weights = '/lustre/yixi/face_segmentation_finetune/gcfv_flow/modelflow/snapshots_gcfvshuffle200200flow/vgg_convflow_xavier_lr1e-12_6000_1e-11/_iter_8000.caffemodel'
#base_weights = '/lustre/yixi/face_segmentation_finetune/gcfv_flow/modelflow/snapshots_gcfvshuffle200200flow/vgg_convflow_xavier_lr1e-12_6000_1e-11_8000_1e-10/_iter_9000.caffemodel'

#Camvidflow finetune
#from /lustre/yixi/face_segmentation_finetune/flow/modelflownp/snapshots_camvidfmp200200epicflow/modeldefaultflowsurg_lr1e-10/_iter_38000.caffemodel
#base_weights = 'camvidflow_surg.caffemodel'
base_weights = 'snapshots_gcfvfmp200200flow/camvidflow_lr1e-10/_iter_24000.caffemodel'


# init
caffe.set_mode_gpu()
caffe.set_device(0)

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(base_weights)


# do net surgery to set the deconvolution weights for bilinear interpolation
#interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
#interp_surgery(solver.net, interp_layers)


#print '----- yixi initialized params ----'
#layernames = solver.net.params.keys()
#for l in layernames:
#	print 'yixi', l, np.mean(solver.net.params[l][0].data), np.max(solver.net.params[l][0].data), np.mean(solver.net.params[l][0].data)



# solve straight through -- a better approach is to define a solving loop to
# 1. take SGD steps
# 2. score the model by the test net `solver.test_nets[0]`
# 3. repeat until satisfied
solver.step(80000)
