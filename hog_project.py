import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def hog(img):
    # Code from OpenCV examples
    bin_n = 16 # Number of bins
    grid_n = (5,4)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = []
    mag_cells = []

    rows, cols = img.shape[:2]
    for i in xrange(grid_n[0]):
        for j in xrange(grid_n[1]):
            r_low = i*rows/grid_n[0]
            r_high = (i+1)*rows/grid_n[1]
            c_low = j*cols/grid_n[1]
            c_high = (j+1)*cols/grid_n[1]
            bin_cells.append(bins[r_low:r_high, c_low:c_high])
            mag_cells.append(mag[r_low:r_high, c_low:c_high])
    # bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    # mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist / np.linalg.norm(hist)

def load_hog():
    directory = 'pointing_images'
    hog_list = []
    for i in xrange(9):
        filepath = os.path.join(os.getcwd(), directory, str(i) + '.JPG')
        print filepath
        im = cv2.imread(filepath)
        hog_list.append(hog(im))
        plt.plot(hog_list[-1])
    plt.show()
    return hog_list

def load_hog_2():
    pass

def closest(frame_hog, hog_list):
    diff_mag = [np.linalg.norm(frame_hog - l) for l in hog_list]
    print(diff_mag)
    return np.argmin(diff_mag)


cap = cv2.VideoCapture(-1)

# for x in xrange(0,512,stepsize):
#     for y in xrange(0,512,stepsize):
hog_list = load_hog()
while True:
    ret, frame = cap.read()

    frame_hog = hog(frame)
    print closest(frame_hog, hog_list)
    cv2.imshow('frame',frame)
    if (cv2.waitKey(30) & 0xFF) == ord('q'):
        broken = True
        break

cap.release()
cv2.destroyAllWindows()