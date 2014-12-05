import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# set this to the location of caffe (instructions for set up are here: http://caffe.berkeleyvision.org/installation.html)
caffe_root = os.path.normpath(os.path.join(os.getcwd(), '../caffe'))
print os.getcwd()
print caffe_root
import sys
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

# set plot parameters

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# setup cnn

net = caffe.Classifier(os.path.join(caffe_root, 'models/bvlc_reference_caffenet/deploy.prototxt'),
                       os.path.join(caffe_root, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'))
net.set_phase_test()
net.set_mode_cpu()
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
net.set_mean('data', np.load(os.path.join(caffe_root, 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))) # ImageNet mean
net.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# function that calculates a feature vector for an image

def calculate_features(img):
    scores = net.predict([img])

    feat = net.blobs['fc7'].data[4]
    feat = np.squeeze(feat)
    return feat

# start capture from webcam

cap = cv2.VideoCapture(-1)

# training

stepsize = 20

images = []
inputs = []
outputs = []

broken = False
for x in xrange(0,512,stepsize):
    for y in xrange(0,512,stepsize):

        ret, frame = cap.read()

        images.append(frame)

        # calculate features
        '''
        feat = calculate_features(frame)
        inputs.append(feat)
        '''
        outputs.append([x,y])


        # show display
        resized = cv2.resize(frame, (0,0), fx=.25, fy=.25)
        height, width = resized.shape[:2]
        img = np.zeros((512,512,3), np.uint8)
        cv2.circle(img,(x,y),10,(255,0,0),-1)
        img[-height:,-width:] = resized

        cv2.imshow('frame',img)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            broken = True
            break

    if broken:
        break

for i in xrange(5):
    feat = calculate_features(images[i*50])
    inputs.append(feat)
    plt.subplot(5,1,i)
    plt.imshow(images[i*50])
plt.show()

inputs_matrix = np.zeros((len(inputs), 4096))
for ind, input in enumerate(inputs):
    inputs_matrix[ind,:] = input

print inputs_matrix.size

mean_vector = np.mean(inputs_matrix,axis=0)

plt.plot(mean_vector)
plt.show()

# show plots of features



for i, feat in enumerate(inputs):
    feat = np.subtract(feat,mean_vector)

    plt.subplot(5,1,i)
    plt.plot(feat)
plt.show()

# create classifier

neigh = KNeighborsClassifier(n_neighbors=3, weights='distance')
print np.vstack(inputs).shape
print np.vstack(outputs)
neigh.fit(np.vstack(inputs), np.vstack(outputs))

# test

while True:
    ret, frame = cap.read()

    feat = calculate_features(frame)

    print neigh.kneighbors(feat)
    location = np.squeeze(neigh.predict(feat))
    print location
    x = location[0]
    y = location[1]

    # show display
    resized = cv2.resize(frame, (0,0), fx=.25, fy=.25)
    height, width = resized.shape[:2]
    img = np.zeros((512,512,3), np.uint8)
    cv2.circle(img,(x,y),10,(255,0,0),-1)
    img[-height:,-width:] = resized

    cv2.imshow('frame',img)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()