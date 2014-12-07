import numpy as np
import cv2

stepsize = 40
pause = 30 # the amount of frames we pause between lines

cap = cv2.VideoCapture(-1)

while True:
    ret, frame = cap.read()

    # show display
    resized = cv2.resize(frame, (0,0), fx=.25, fy=.25)
    height, width = resized.shape[:2]
    img = np.zeros((512,512,3), np.uint8)
    img[-height:,-width:] = resized

    cv2.imshow('frame',img)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

broken = False
for x in xrange(0,512,stepsize):
    for i in xrange(pause):
        ret, frame = cap.read()

        # show display
        resized = cv2.resize(frame, (0,0), fx=.25, fy=.25)
        height, width = resized.shape[:2]
        img = np.zeros((512,512,3), np.uint8)
        cv2.circle(img,(x,0),10,(255,0,0),-1)
        img[-height:,-width:] = resized

        cv2.imshow('frame',img)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            broken = True
            break

    for y in xrange(0,512,stepsize):
        ret, frame = cap.read()
        cv2.imwrite('clean_data_2/hand0_%d_%d.png' % (x,y), frame)

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
