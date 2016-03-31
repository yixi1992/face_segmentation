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
# base_weights = 'vgg16fc.caffemodel'
# base_weights = '../VGG16fc.caffemodel'
# base_weights = '/lustre/yixi/face_segmentation_finetune/fullconv/model2/snapshots_camvid300/train_lr1e-12/_iter_12300.caffemodel'
# base_weights = '/lustre/yixi/face_segmentation_finetune/fullconv/model2/snapshots_camvid300/train_lr1e-12_12300_lr1e-10_12500/_iter_13000.caffemodel'
# base_weights = '/lustre/yixi/face_segmentation_finetune/fcn8/model1/snapshots_camvid300/train_lr1e-10w0.0005/_iter_12000.caffemodel'
# base_weights = '/lustre/yixi/face_segmentation_finetune/fcn8/model1/snapshots_camvid300/train_lr1e-10w0.0005_12000/_iter_26000.caffemodel'
#base_weights = '/lustre/yixi/face_segmentation_finetune/flow/model1/snapshots_camvid300flow/train_lr1e-10/_iter_4850.caffemodel'
#base_weights = '/lustre/yixi/face_segmentation_finetune/flow/modelflownp/snapshots_camvid200flow/train_lr1e-10/_iter_77000.caffemodel'
#base_weights = '/lustre/yixi/face_segmentation_finetune/flow/modelflownp/snapshots_camvid200flow/train_lr1e-10_19000_1e-12/_iter_77000.caffemodel'

#base_weights = '/lustre/yixi/face_segmentation_finetune/flow/modelflownp/snapshots_camvid200flow/train_lr1e-10_19000/_iter_4000.caffemodel'
#base_weights = '/lustre/yixi/face_segmentation_finetune/flow/modelflownp/snapshots_camvid200flow/train_lr1e-10_19000_4000_1e-12/_iter_58000.caffemodel'

#base_weights = '/lustre/yixi/face_segmentation_finetune/flow/modelflownp/snapshots_camvid200flow/train_lr1e-10_19000_4000_1e-12_58000/_iter_58000.caffemodel'
base_weights = '/lustre/yixi/face_segmentation_finetune/flow/modelflownp/snapshots_camvid200flow/train_lr1e-10_19000_4000_1e-12_58000_58000/_iter_77000.caffemodel'

# init
caffe.set_mode_gpu()
caffe.set_device(0)

solver = caffe.SGDSolver('solver.prototxt')

# do net surgery to set the deconvolution weights for bilinear interpolation
#interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
#interp_surgery(solver.net, interp_layers)

# copy base weights for fine-tuning
solver.net.copy_from(base_weights)

# solve straight through -- a better approach is to define a solving loop to
# 1. take SGD steps
# 2. score the model by the test net `solver.test_nets[0]`
# 3. repeat until satisfied
solver.step(80000)