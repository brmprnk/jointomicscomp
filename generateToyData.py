import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit

np.random.seed(1991)

# example 1: 2 classes in 3D
Ntr = 10000
N = 2 * Ntr
d = 3

# view 1: uniform in sphere
# class 0 in radius [0,2], class 1 in radius [3,4]

r1 = 2 * np.random.rand(N)
theta1 = np.pi * np.random.rand(N)
phi1 = 2 * np.pi * np.random.rand(N)

r2 = 3 + np.random.rand(N)
theta2 = np.pi * np.random.rand(N)
phi2 = 2 * np.pi * np.random.rand(N)

X1 = np.vstack((r1 * np.sin(theta1) * np.cos(phi1), r1 * np.sin(theta1) * np.sin(phi1), r1 * np.cos(theta1)))
X2 = np.vstack((r2 * np.sin(theta2) * np.cos(phi2), r2 * np.sin(theta2) * np.sin(phi2), r2 * np.cos(theta2)))

X = np.hstack((X1, X2)).T

# class labels
y = np.ones(X.shape[0], int)
y[:N] = 0

cv = StratifiedShuffleSplit(n_splits=2, test_size=0.5)

for train_ind, test_ind in cv.split(X, y):
	v1train = X[train_ind]
	ytrain = y[train_ind]
	v1test = X[test_ind]
	ytest = y[test_ind]


# view 2: overlapping, spherical gaussians
m1 = np.zeros(d)
m2 = np.ones(d) * 1.5

C = np.eye(d)
X1  = np.random.multivariate_normal(m1, C, N)
X2  = np.random.multivariate_normal(m2, C, N)

X = np.vstack((X1, X2))
v2train = X[train_ind]
v2test = X[test_ind]

clf1 = KNeighborsClassifier(n_neighbors=5).fit(v1train, ytrain)
print('View 1 error: %.3f' % (1 - clf1.score(v1test, ytest)))

clf2 = KNeighborsClassifier(n_neighbors=5).fit(v2train, ytrain)
print('View 2 error: %.3f' % (1 - clf2.score(v2test, ytest)))

np.save('artificial-data/example-1/view1_train.npy', v1train)
np.save('artificial-data/example-1/view2_train.npy', v2train)
np.save('artificial-data/example-1/y_train.npy', ytrain)

np.save('artificial-data/example-1/view1_test.npy', v1test)
np.save('artificial-data/example-1/view2_test.npy', v2test)
np.save('artificial-data/example-1/y_test.npy', ytest)
