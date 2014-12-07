import numpy as np
import cv2
import matplotlib.pyplot as plt

import cnn

import pickle

inputs = []
outputs = []
stepsize = 40

for file_start in ['hand', 'hand2', 'hand3']:
    for x in xrange(0,512,stepsize):
        print x
        for y in xrange(0,512,stepsize):

            image = cnn.caffe.io.load_image('clean_data/%s_%i_%i.png' % (file_start, x,y))

            feat = cnn.calculate_features(image)
            inputs.append(feat)
            outputs.append((x,y))

pickle.dump((inputs, outputs), open('features_clean.p', 'wb'))

print 'done training'

'''
to make images loaded with opencv work, we have to do this:
img = cv2.imread('/home/david/PycharmProjects/819 project/data/hand_0_0.png')
img = img[:,:,[2,1,0]]
img = img * 1.0/255
'''