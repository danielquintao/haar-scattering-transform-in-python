"""This is an illustration of classification in MNIST: original versus Haar Scattering features

We compare the representations in both a nonlinear and a linear classifiers.
We did NOT perform dimension reduction as in Chen X., Cheng X., Mallat S. 2014

"""

from read_MNIST import read_10000_from_MNIST
import numpy as np
from haar_scattering_transform import HaarScatteringTransform
from unstructured2graphs import graph_for_grid
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV

X_train, y_train = read_10000_from_MNIST()
X_test, y_test = read_10000_from_MNIST(train=False)
X_train = X_train.reshape(-1, 28 * 28)
X_test = X_test.reshape(-1, 28 * 28)

haar = HaarScatteringTransform(graph_for_grid((28, 28)), J=3)

X_train_haar = []
for X in X_train[:]:
    transform = haar.get_haar_scattering_transform(X)
    X_train_haar.append(transform[-1].flatten())
X_train_haar = np.array(X_train_haar)
print(X_train_haar.shape)

X_test_haar = []
for X in X_test[:]:
    transform = haar.get_haar_scattering_transform(X)
    X_test_haar.append(transform[-1].flatten())
X_test_haar = np.array(X_test_haar)
print(X_test_haar.shape)

# CLASSIFICATION

# Random Forest
params = {'max_depth': [2, 3, 5, 10]}
clf = GridSearchCV(RandomForestClassifier(), param_grid=params, cv=3, scoring="accuracy", refit=False)
clf.fit(X_train_haar[:1000], y_train[:1000])  # grid search on less samples
clf = RandomForestClassifier(**clf.best_params_)
clf.fit(X_train_haar, y_train)  # refit on whole train dataset
print("SCORE USING HAAR FEATURES (Random Forest Classifier)", clf.score(X_test_haar, y_test))
clf = GridSearchCV(RandomForestClassifier(), param_grid=params, cv=3, scoring="accuracy", refit=False)
clf.fit(X_train[:1000], y_train[:1000])  # grid search on less samples
clf = RandomForestClassifier(**clf.best_params_)
clf.fit(X_train, y_train)  # refit on whole train dataset
print("SCORE USING ORIGINAL (FLATTENED) FEATURES (Random Forest Classifier)", clf.score(X_test, y_test))

# # Linear SVC
# clf = make_pipeline(StandardScaler(),
#                     LinearSVC(random_state=0, tol=1e-5))
# clf.fit(X_train_haar, y_train)
# print("SCORE USING HAAR FEATURES (Random Forest Classifier)", clf.score(X_test_haar, y_test))
# clf = make_pipeline(StandardScaler(),
#                     LinearSVC(random_state=0, tol=1e-5))
# clf.fit(X_train, y_train)
# print("SCORE USING ORIGINAL (FLATTENED) FEATURES (Random Forest Classifier)", clf.score(X_test, y_test))

# (a simple classifier) Ridge
clf = RidgeClassifier()
clf.fit(X_train_haar, y_train)
print("SCORE USING HAAR FEATURES (Ridge Classifier)", clf.score(X_test_haar, y_test))
clf.fit(X_train, y_train)
print("SCORE USING ORIGINAL (FLATTENED) FEATURES (Ridge Classifier)", clf.score(X_test, y_test))
