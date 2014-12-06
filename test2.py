import numpy as np
import pickle
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt

(inputs, outputs) = pickle.load(open('features_clean.p', 'rb'))
n = len(inputs)
train = 2*n/3
third = n/3
train_inputs = inputs[third:]
train_outputs = outputs[third:]
test_inputs = inputs[:third]
test_outputs = outputs[:third]

stack_inputs = np.vstack(train_inputs)
stack_outputs = np.vstack(train_outputs)
neigh = KNeighborsClassifier(n_neighbors=3, weights='distance')
neigh.fit(stack_inputs, stack_outputs)

regression = RidgeCV()
regression.fit(stack_inputs, stack_outputs)

neighbors_distances = []
regression_distances = []
middle_distances = []

plt.axis([0,512,0,512])

for i in xrange(len(test_inputs)):
    feat = test_inputs[i]
    (x,y) = test_outputs[i]
    print x, y
    loc_neigh = np.squeeze(neigh.predict(feat))
    print 'neighbors ', loc_neigh
    loc_reg = regression.predict(feat)
    print 'regression ', loc_reg

    neighbors_distances.append(math.hypot(x-loc_neigh[0], y-loc_neigh[1]))
    regression_distances.append(math.hypot(x-loc_reg[0], y-loc_reg[1]))
    middle_distances.append(math.hypot(x-256,y-256))

    plt.arrow(x,y,loc_neigh[0]-x,loc_neigh[1]-y)

plt.show()

plt.subplot(3,1,1)
plt.plot(neighbors_distances)
plt.subplot(3,1,2)
plt.plot(regression_distances)
plt.subplot(3,1,3)
plt.plot(middle_distances)
plt.show()