import numpy as np
import cv2

stepsize = 80#40
pause = 20 # the amount of frames we pause between lines

cap = cv2.VideoCapture(-1)

while True:
    ret, frame = cap.read()

    # show display
    resized = cv2.resize(frame, (0,0), fx=512.0/frame.shape[1],fy=512.0/frame.shape[0])#fx=.25, fy=.25)
    height, width = resized.shape[:2]
    img = np.zeros((512,512,3), np.uint8)
    img[-height:,-width:] = cv2.flip(resized,1)

    cv2.imshow('frame',img)
    if (cv2.waitKey(1) & 0xFF) == ord(' '):
        break

broken = False
for x in xrange(0,512,stepsize):
    # for i in xrange(pause):
    #     ret, frame = cap.read()
    #
    #     # show display
    #     resized = cv2.resize(frame, (0,0), fx=.25, fy=.25)
    #     height, width = resized.shape[:2]
    #     img = np.zeros((512,512,3), np.uint8)
    #     cv2.circle(img,(x,0),10,(255,0,0),-1)
    #     img[-height:,-width:] = resized
    #
    #     cv2.imshow('frame',img)
    #     if (cv2.waitKey(1) & 0xFF) == ord('q'):
    #         broken = True
    #         break

    for y in xrange(0,512,stepsize):
        while True:
            ret, frame = cap.read()
            # show display
            resized = cv2.resize(frame, (0,0), fx=512.0/frame.shape[1],fy=512.0/frame.shape[0])#fx=.25, fy=.25)
            height, width = resized.shape[:2]
            img = np.zeros((512,512,3), np.uint8)
            img[-height:,-width:] = cv2.flip(resized,1)
            cv2.circle(img,(x,y),10,(255,0,0),-1)
            cv2.line(img,(0,img.shape[1]/2), (img.shape[0]-1,img.shape[1]/2),(255,255,255))
            cv2.line(img,(img.shape[0]/2,0), (img.shape[0]/2,img.shape[1]-1),(255,255,255))
            cv2.imshow('frame',img)
            key = cv2.waitKey(5) & 0xFF;
            if (key == ord('q')):
                broken = True
                break
            elif (key == ord(' ')):
                break
        if broken:
            break
        cv2.imwrite('clean_data_close/hand14_%d_%d.png' % (x,y), frame)

    if broken:
        break
