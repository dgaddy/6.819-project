import numpy as np
import cv2
import os, math
import pickle
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeCV

class Smoother:
    smoothing = 0.5
    outlier_thresh = 200
    def __init__(self):
        self.smoothed = (0, 0)
        self.hist = [(0, 0)]
        self.hist_len = 5

    def add_history(self, val):
        self.hist.append(val)
        if len(self.hist) > self.hist_len:
            self.hist.pop(0)

    def mean_tuple(self, tuple_list):
        return tuple(sum(x) / len(x) for x in zip(*tuple_list))

    def add(self, val):
        mean = self.mean_tuple(self.hist)
        delta_norm = np.linalg.norm([val[i]-mean[i] for i in range(len(val))])
        self.add_history(val)
        if delta_norm < Smoother.outlier_thresh:
            print val
            self.smoothed = tuple(self.smoothed[i]*Smoother.smoothing + (1-Smoother.smoothing)*val[i] for i in xrange(len(val)))
        else:
            print "Didn't add"
        return self.smoothed

    def get(self):
        return self.smoothed

def hog(img, asHlist=True):
    # Code from OpenCV examples
    bin_n = 9 # Number of bins
    #grid_n = (12,20)
    grid_n = (8,12)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = []
    mag_cells = []

    rows, cols = img.shape[:2]
    for i in xrange(grid_n[0]):
        for j in xrange(grid_n[1]):
            rs = slice(i*rows/grid_n[0], (i+1)*rows/grid_n[0])
            cs = slice(j*cols/grid_n[1], (j+1)*cols/grid_n[1])
            bin_cells.append(bins[rs, cs])
            mag_cells.append(mag[rs, cs])
    # bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    # mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    # [[bin1magSum,bin2magSum,...bin9magSum], [bin1magSum,bin2magSum,...,bin9magSum]]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    if asHlist:
        return hist / np.linalg.norm(hist)
    else:
        hists = np.reshape(hists,grid_n + (bin_n,))
        return hists / np.linalg.norm(hist)

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
    directory = 'clean_data_close'
    hog_list = []
    coordinates = []
    for filepath in os.listdir(os.path.join(os.getcwd(), directory)):
        if not filepath.endswith('.png'):
            continue
        im = cv2.imread(os.path.join(os.getcwd(), directory, filepath))
        hog_list.append(hog(im))
        coordinates.append([int(n) for n in os.path.splitext(os.path.basename(filepath))[0].split('_') if n.isdigit()])
    return hog_list, coordinates

def closest(frame_hog, hog_list):
    diff_mag = [np.linalg.norm(frame_hog - l) for l in hog_list]
    return np.argmin(diff_mag)

def hogPicture(w,bs):
    # Make picture of positive HOG weights.
    s = w.shape

    # construct a "glyph" for each orientaion
    bim1 = np.zeros((bs, bs));
    center_col = math.floor(bs/2)
    bim1[:,center_col:center_col+1] = 1;
    bim = []
    bim.append(bim1)
    for i in xrange(1,bs):
        rot_mat = cv2.getRotationMatrix2D((center_col, center_col), -i*20, 1)
        bim.append(cv2.warpAffine(bim1,rot_mat,(bs,bs)))

    # make pictures of positive weights bs adding up weighted glyphs
    #w(w < 0) = 0; # Don't think we need this
    im = np.zeros((bs*s[0], bs*s[1]))
    for i in xrange(s[0]):
        iis = slice(i*bs, (i+1)*bs)
        for j in xrange(0, s[1]):
            jjs = slice(j*bs, (j+1)*bs)
            for k in xrange(s[2]):
                im[iis,jjs] = im[iis,jjs] + bim[k] * w[i,j,k] * 2 #scale for brightness
    return im


def main():
    cap = cv2.VideoCapture(-1)

    smoother = Smoother()
    hog_list, coords = pickle.load(open('hog_features_clean.p', 'rb'))#load_hog_2()
    reg = RidgeCV()
    reg.fit(np.vstack(hog_list), np.vstack(coords))
    neigh = KNeighborsClassifier(n_neighbors=5, weights='distance')
    neigh.fit(np.vstack(hog_list), np.vstack(coords))
    #video  = cv2.VideoWriter('video.avi', cv2.cv.FOURCC(*'XVID'), 12, (640, 480))
    #print video.isOpened()
    while True:
        ret, frame = cap.read()

        frame_hog = hog(frame, asHlist=False)
        frame_hog_hlist = hog(frame)
        reg_loc = np.squeeze(reg.predict(np.hstack(frame_hog_hlist)))
        neigh_loc = np.squeeze(neigh.predict(np.hstack(frame_hog_hlist)))
        location = smoother.add(tuple((reg_loc[i] + neigh_loc[i])/2 for i in xrange(len(neigh_loc))))
        location = (location[0]*frame.shape[1]/512.0, location[1]*frame.shape[0]/512.0)
        print location

        frame = cv2.flip(frame, 1)
        cv2.circle(frame,tuple(int(n) for n in location),10,(255,0,0),-1)
        fw = frame.shape[0]
        fh = frame.shape[1]
        cv2.line(frame, (0, fw/2), (fh, fw/2), (0, 0, 0))
        cv2.line(frame, (fh/2, 0), (fh/2, fw), (0, 0, 0))
        cv2.imshow('frame', frame)
        #video.write(frame)

        if (cv2.waitKey(30) & 0xFF) == ord('q'):
            break

        img = hogPicture(frame_hog, 40)
        cv2.imshow('hog',img)

    cap.release()
    #video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # loaded_hog = load_hog_2()
    # pickle.dump(loaded_hog, open('hog_features_clean.p', 'wb'))
    # print "Hog inputs pickled"
    main()