import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit


def example1(Ntr, d, seed=1991):

	np.random.seed(seed)
	N = 2 * Ntr

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

	v1 = np.hstack((X1, X2)).T

	# class labels
	y = np.ones(v1.shape[0], int)
	y[:N] = 0

	# view 2: overlapping, spherical gaussians
	m1 = np.zeros(d)
	m2 = np.ones(d) * 1.5

	C = np.eye(d)
	X1  = np.random.multivariate_normal(m1, C, N)
	X2  = np.random.multivariate_normal(m2, C, N)

	v2 = np.vstack((X1, X2))

	return v1, v2, y

####################################################
# example 2
####################################################

def example2(Ntr, d, seed=1991):
	# 1 informative feature, d-1 noise
	assert d > 1

	np.random.seed(seed)
	N = 2 * Ntr

	# common feature is first in view1, 2nd in view2
	cInd1 = 0
	cInd2 = 1

	# common feature mean and variance
	m1 = 0
	m2 = 3
	s = 1

	# noise variance
	sn = 1e-10

	# view 1:
	X1 = np.zeros((N, d))
	X2 = np.zeros(X1.shape)

	# class labels
	y = np.ones(2*N, int)
	y[:N] = 0

	for i in range(d):
		if i == cInd1:
			X1[:, i] = np.random.normal(m1, s, size=X1.shape[0])
			X2[:, i] = np.random.normal(m2, s, size=X1.shape[0])
		else:
			# uniform in -5, 5
			X1[:, i] = 10 * np.random.rand(N) - 5
			X2[:, i] = 10 * np.random.rand(N) - 5

	v1 = np.vstack((X1, X2))

	# view 2:
	X1n = np.zeros((N, d))
	X2n = np.zeros(X1n.shape)


	for i in range(d):
		if i == cInd2:
			X1n[:, i] = np.random.normal(X1[:, cInd1], sn)
			X2n[:, i] = np.random.normal(X2[:, cInd1], sn)
		else:
			# uniform in -5, 5
			X1n[:, i] = 10 * np.random.rand(N) - 5
			X2n[:, i] = 10 * np.random.rand(N) - 5

	v2 = np.vstack((X1n, X2n))

	return v1, v2, y




ex = sys.argv[1]

# example 1: 2 classes in 3D
Ntr = 10000
d = 3

v1, v2, y = eval('example' + ex + '(Ntr, d)')

cv = StratifiedShuffleSplit(n_splits=2, test_size=0.5)

for train_ind, test_ind in cv.split(v1, y):
	v1train = v1[train_ind]
	v2train = v2[train_ind]
	ytrain = y[train_ind]
	v1test = v1[test_ind]
	ytest = y[test_ind]
	v2test = v2[test_ind]



clf1 = KNeighborsClassifier(n_neighbors=5).fit(v1train, ytrain)
print('View 1 error: %.3f' % (1 - clf1.score(v1test, ytest)))

clf2 = KNeighborsClassifier(n_neighbors=5).fit(v2train, ytrain)
print('View 2 error: %.3f' % (1 - clf2.score(v2test, ytest)))


np.save('artificial-data/example-' + ex + '/view1_train.npy', v1train)
np.save('artificial-data/example-' + ex + '/view2_train.npy', v2train)
np.save('artificial-data/example-' + ex + '/y_train.npy', ytrain)

np.save('artificial-data/example-' + ex + '/view1_test.npy', v1test)
np.save('artificial-data/example-' + ex + '/view2_test.npy', v2test)
np.save('artificial-data/example-' + ex + '/y_test.npy', ytest)
