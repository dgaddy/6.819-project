import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

import cnn

# set plot parameters

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
# function that calculates a feature vector for an image

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
    feat = cnn.calculate_features(images[i*50])
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

    feat = cnn.calculate_features(frame)

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