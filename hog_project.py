import numpy as np
import cv2
import matplotlib.pyplot as plt

def hog(img):
    SZ=20
    bin_n = 16 # Number of bins
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist

cap = cv2.VideoCapture(-1)

# for x in xrange(0,512,stepsize):
#     for y in xrange(0,512,stepsize):
while True:
    ret, frame = cap.read()

    plt.ion()
    plt.clf()
    plt.plot(hog(frame))
    # plt.show(block=False)
    cv2.imshow('frame',frame)
    if (cv2.waitKey(30) & 0xFF) == ord('q'):
        broken = True
        break

cap.release()
cv2.destroyAllWindows()