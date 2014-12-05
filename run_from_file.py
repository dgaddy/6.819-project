import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# set this to the location of caffe (instructions for set up are here: http://caffe.berkeleyvision.org/installation.html)
caffe_root = '/home/david/workspace/caffe/'
import sys
sys.path.insert(0, caffe_root+'python')
import caffe

net = caffe.Classifier(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                       caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
net.set_phase_test()
net.set_mode_cpu()
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
net.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))  # ImageNet mean
net.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# function that calculates a feature vector for an image

def calculate_features(img):
    scores = net.predict([img])

    feat = net.blobs['fc7'].data[4]
    feat = np.squeeze(feat)
    return feat

inputs = []
outputs = []
stepsize = 20

test = [(100,100),(100,200),(200,200)]

for x in xrange(0,512,stepsize):
    for y in xrange(0,512,stepsize):
        if (x,y) in test:
            continue

        image = caffe.io.load_image('/home/david/PycharmProjects/819 project/data/hand_%i_%i.png' % (x,y))

        feat = calculate_features(image)
        inputs.append(feat)
        outputs.append((x,y))

neigh = KNeighborsClassifier(n_neighbors=3, weights='distance')
neigh.fit(np.vstack(inputs), np.vstack(outputs))

for (x,y) in test:
    image = caffe.io.load_image('/home/david/PycharmProjects/819 project/data/hand_%i_%i.png' % (x,y))
    feat = calculate_features(image)
    print (x,y)
    print neigh.kneighbors(feat)
    location = np.squeeze(neigh.predict(feat))
    print location
'''
to make images loaded with opencv work, we have to do this:
img = cv2.imread('/home/david/PycharmProjects/819 project/data/hand_0_0.png')
img = img[:,:,[2,1,0]]
img = img * 1.0/255
'''