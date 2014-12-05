import pickle
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier

import cnn

(inputs, outputs) = pickle.load(open('features.p', 'rb'))
stack_inputs = np.vstack(inputs)

mean_vector = np.mean(stack_inputs,axis=0)

stack_inputs = stack_inputs - mean_vector

neigh = KNeighborsClassifier(n_neighbors=3, weights='distance')
neigh.fit(stack_inputs, np.vstack(outputs))

cap = cv2.VideoCapture(-1)


ret, frame = cap.read()

background = cnn.calculate_features(frame)
while True:
    ret, frame = cap.read()

    feat = cnn.calculate_features(frame)
    feat = feat - background

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
