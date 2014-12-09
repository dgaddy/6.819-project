import numpy as np
import pickle
import math
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.neighbors import DistanceMetric
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import hog_project
import ast

(inputs, outputs) = pickle.load(open('features_clean.p', 'rb'))
(hog_inputs, hog_outputs) = pickle.load(open('hog_features_clean.p', 'rb'))
n = len(inputs)
train = 2*n/3
third = n/6
for start in xrange(15):
#start = 14
    test_slice_bounds = (start*n/15, (start+1)*n/15)

    train_slice_1 = slice(None, test_slice_bounds[0]) if test_slice_bounds[0] is not None else slice (0, 0)
    test_slice = slice(test_slice_bounds[0], test_slice_bounds[1])
    train_slice_2 = slice(test_slice_bounds[1], None) if test_slice_bounds[1] is not None else slice(0, 0)

    train_inputs = inputs[train_slice_1] + inputs[train_slice_2]
    train_outputs = outputs[train_slice_1] + outputs[train_slice_2]
    hog_train_inputs = hog_inputs[train_slice_1] + hog_inputs[train_slice_2]
    hog_train_outputs = hog_outputs[train_slice_1] + hog_outputs[train_slice_2]
    test_inputs = inputs[test_slice]
    test_outputs = outputs[test_slice]
    hog_test_inputs = hog_inputs[test_slice]
    hog_test_outputs = hog_outputs[test_slice]

    stack_inputs = np.vstack(train_inputs)
    stack_outputs = np.vstack(train_outputs)

    #pca = PCA(n_components=10)
    #pca.fit(hog_train_inputs)
    #hog_train_inputs = pca.transform(hog_train_inputs)

    #hog_test_inputs = pca.transform(hog_test_inputs)

    #stack_inputs = pca.transform(stack_inputs)
    #print stack_inputs.shape

    #mean = stack_inputs.mean(axis=0)
    #std = stack_inputs.std(axis=0)
    #std = std + (std < 1e-3) # if std is 0, set it to 1 to avoid infinities
    #stack_inputs = stack_inputs - mean
    #stack_inputs = stack_inputs / std

    neigh = KNeighborsRegressor(n_neighbors=4, weights='distance')
    neigh.fit(stack_inputs, stack_outputs)

    stack_inputs_poly = np.concatenate((stack_inputs, stack_inputs**2), 1)
    regression = LinearRegression()
    regression.fit(stack_inputs, stack_outputs)
    '''reg2_x = SGDRegressor()
    reg2_x.fit(stack_inputs, stack_outputs[:,0])
    reg2_y = SGDRegressor()
    reg2_y.fit(stack_inputs, stack_outputs[:,1])'''

    neigh_hog = KNeighborsRegressor(n_neighbors=5, weights='distance')
    neigh_hog.fit(np.vstack(hog_train_inputs), np.vstack(hog_train_outputs))

    regression_hog = LinearRegression()
    regression_hog.fit(np.vstack(hog_train_inputs), np.vstack(hog_train_outputs))

    svc_hog = SVC()
    svc_hog.fit(np.vstack(hog_train_inputs), tuple(str(c) for c in hog_train_outputs))
    print svc_hog.score(np.vstack(hog_train_inputs), tuple(str(c) for c in hog_train_outputs))

    try:
        neighbors_distances
    except NameError:
        neighbors_distances = []
        regression_distances = []
        middle_distances = []
        hog_distances = []
        hog_reg_distances = []
        hog_svc_distances = []
        hog_nn_reg_distances = []

    plt.axis([0,512,0,512])

    for i in xrange(len(test_inputs)):
        feat = test_inputs[i]
        (x,y) = test_outputs[i]
        (xhog,yhog) = hog_test_outputs[i]
        # print x, y
        #feat_normalized = (feat - mean)/std
        #feat = pca.transform(feat)
        #print feat.shape
        loc_neigh = np.squeeze(neigh.predict(feat))
        #print loc_neigh
        # print 'neighbors ', loc_neigh
        #feat_poly = np.concatenate((feat, feat**2))

        loc_reg = regression.predict(feat)
        #reg_x = reg2_x.predict(feat)
        #reg_y = reg2_y.predict(feat)
        #loc_reg = (reg_x,reg_y)
        # print 'regression ', loc_reg
        loc_hog = np.squeeze(neigh_hog.predict(hog_test_inputs[i]))
        loc_hog_reg = np.squeeze(regression_hog.predict(hog_test_inputs[i]))
        loc_hog_svc = np.squeeze(ast.literal_eval(svc_hog.predict(hog_test_inputs[i])[0]))
        loc_hog_nn_reg = tuple((loc_hog[i] + loc_hog_reg[i])/2.0 for i in xrange(len(loc_hog)))

        neighbors_distances.append(math.hypot(x-loc_neigh[0], y-loc_neigh[1]))
        regression_distances.append(math.hypot(x-loc_reg[0], y-loc_reg[1]))
        middle_distances.append(math.hypot(x-256,y-256))
        hog_distances.append(math.hypot(xhog-loc_hog[0], yhog-loc_hog[1]))
        hog_reg_distances.append(math.hypot(xhog-loc_hog_reg[0], yhog-loc_hog_reg[1]))
        hog_svc_distances.append(math.hypot(xhog-loc_hog_svc[0], yhog-loc_hog_svc[1]))
        hog_nn_reg_distances.append(math.hypot(xhog-loc_hog_nn_reg[0], yhog-loc_hog_nn_reg[1]))

        #plt.arrow(x,y,loc_neigh[0]-x,loc_neigh[1]-y)
        #plt.arrow(xhog,yhog,loc_hog_nn_reg[0]-xhog,loc_hog_nn_reg[1]-yhog, length_includes_head=True, head_width=5)
        #plt.draw()

    #plt.show()

print '\nAverage Errors'
print 'CNN Middle: ', sum(middle_distances) / len(middle_distances)
print 'CNN Neighbors: ', sum(neighbors_distances) / len(neighbors_distances)
print 'CNN Regression: ', sum(regression_distances) / len(regression_distances)

print 'HOG Neighbors: ', sum(hog_distances) / len(hog_distances)
print 'HOG Regression', sum(hog_reg_distances) / len(hog_reg_distances)
#print 'HOG SVC', sum(hog_svc_distances) / len(hog_svc_distances)
print 'HOG NN Reg', sum(hog_nn_reg_distances) / len(hog_nn_reg_distances)

subplots = (5,1)
#plt.subplot(subplots[0],subplots[1],1)
#plt.hist(middle_distances)#plt.plot(middle_distances)
plt.subplot(subplots[0],subplots[1],1)
plt.hist(neighbors_distances,bins=20,range=(0,250))#plt.plot(neighbors_distances)
plt.title('CNN KNN')
plt.ylabel('Error Count')

plt.subplot(subplots[0],subplots[1],2)
plt.hist(regression_distances,bins=20,range=(0,250))#plt.plot(regression_distances)
plt.title('CNN Regression')
plt.ylabel('Error Count')


plt.subplot(subplots[0],subplots[1],3)
plt.hist(hog_distances,bins=20,range=(0,250))#plt.plot(hog_distances)
plt.title('HOG KNN')
plt.ylabel('Error Count')

plt.subplot(subplots[0],subplots[1],4)
plt.hist(hog_reg_distances,bins=20,range=(0,250))#plt.plot(hog_reg_distances)
plt.title('HOG Regression')
plt.ylabel('Error Count')

plt.subplot(subplots[0],subplots[1],5)
plt.title('HOG KNN / Regression')
plt.hist(hog_nn_reg_distances,bins=20,range=(0,250))#plt.plot(hog_nn_reg_distances)
plt.ylabel('Error Count')

plt.show()