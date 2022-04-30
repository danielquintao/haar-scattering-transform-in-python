""" This example is an illustration comparing the boolean and original versions of Haar Scattering Transform

"""

from read_MNIST import read_10000_from_MNIST
import numpy as np
from haar_scattering_transform import HaarScatteringTransform
from unstructured2graphs import graph_for_grid
from sklearn.linear_model import RidgeClassifier


X_train, y_train = read_10000_from_MNIST()
X_test, y_test = read_10000_from_MNIST(train=False)
X_train = X_train.reshape(-1, 28 * 28)
X_test = X_test.reshape(-1, 28 * 28)

haar = HaarScatteringTransform(graph_for_grid((28, 28)), J=3)

# ORIGINAL (SCALAR)

X_train_haar = []
for X in X_train[:]:
    transform = haar.get_haar_scattering_transform(X)
    X_train_haar.append(transform[-1].flatten())
X_train_haar = np.array(X_train_haar)

X_test_haar = []
for X in X_test[:]:
    transform = haar.get_haar_scattering_transform(X)
    X_test_haar.append(transform[-1].flatten())
X_test_haar = np.array(X_test_haar)

clf = RidgeClassifier()
clf.fit(X_train_haar, y_train)
print("ORIGINAL (SCALAR) HAAR TRANSFORM (Ridge Classifier):", clf.score(X_test_haar, y_test))

# BOOLEAN

X_train = X_train > 0.5  # convert to bool
X_test = X_test > 0.5  # convert to bool

X_train_haar = []
for X in X_train[:]:
    transform = haar.get_haar_scattering_transform(X)  # automatically recognizes that it is boolean
    X_train_haar.append(transform[-1].flatten())
X_train_haar = np.array(X_train_haar).astype(float)

X_test_haar = []
for X in X_test[:]:
    transform = haar.get_haar_scattering_transform(X)  # automatically recognizes that it is boolean
    X_test_haar.append(transform[-1].flatten())

clf = RidgeClassifier()
clf.fit(X_train_haar, y_train)
print("BOOLEAN HAAR TRANSFORM (Ridge Classifier):", clf.score(X_test_haar, y_test))