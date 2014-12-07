import numpy as np
import cv2
import os, math
import pickle
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

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
    directory = 'clean_data_2'
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

    # for x in xrange(0,512,stepsize):
    #     for y in xrange(0,512,stepsize):
    hog_list, coords = pickle.load(open('hog_features_clean.p', 'rb'))#load_hog_2()
    neigh = KNeighborsClassifier(n_neighbors=5, weights='distance')
    print np.vstack(hog_list).shape
    print np.vstack(coords)
    neigh.fit(np.vstack(hog_list), np.vstack(coords))
    while True:
        ret, frame = cap.read()

        frame_hog = hog(frame, asHlist=False)
        location = np.squeeze(neigh.predict(np.hstack(frame_hog.reshape(-1,9))))
        print location

        #print closest(frame_hog, hog_list)
        #cv2.imshow('frame',frame)
        if (cv2.waitKey(30) & 0xFF) == ord('q'):
            broken = True
            break

        img = hogPicture(frame_hog, 40)
        cv2.imshow('hog',img)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # loaded_hog = load_hog_2()
    # pickle.dump(loaded_hog, open('hog_features_clean.p', 'wb'))
    # print "Hog inputs pickled"
    main()