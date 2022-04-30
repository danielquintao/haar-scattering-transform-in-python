"""Another classification illustration, this time with ECG5000 data set

"""

from read_ECG5000 import read_ECG5000
import numpy as np
from haar_scattering_transform import HaarScatteringTransform
from unstructured2graphs import graph_for_per_time_series
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight

X_train, y_train = read_ECG5000()
X_test, y_test = read_ECG5000(train=False)

haar = HaarScatteringTransform(graph_for_per_time_series(X_train.shape[1]), J=3)

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

# COMPUTE (TEST DATA) SAMPLE WEIGHTS TO BEAT CLASS IMBALANCE
print("class distribution in train data:", np.histogram(y_train, bins=[1, 2, 3, 4, 5, 6])[0])
print("class distribution in test data:", np.histogram(y_test,  bins=[1, 2, 3, 4, 5, 6])[0])
sample_weight_vect = compute_sample_weight(class_weight="balanced", y=y_test)


# CLASSIFICATION

# Random Forest
params = {'max_depth': [2, 3, 5, 10]}
clf = GridSearchCV(RandomForestClassifier(), param_grid=params, cv=2, scoring="accuracy", refit=False)
clf.fit(X_train_haar[:1000], y_train[:1000])  # grid search on less samples
clf = RandomForestClassifier(**clf.best_params_)
clf.fit(X_train_haar, y_train)  # refit on whole train dataset
print("CLASS-BALANCED SCORE USING HAAR FEATURES (Random Forest Classifier)",
      clf.score(X_test_haar, y_test, sample_weight=sample_weight_vect))
clf = GridSearchCV(RandomForestClassifier(), param_grid=params, cv=2, scoring="accuracy", refit=False)
clf.fit(X_train[:1000], y_train[:1000])  # grid search on less samples
clf = RandomForestClassifier(**clf.best_params_)
clf.fit(X_train, y_train)  # refit on whole train dataset
print("CLASS-BALANCED SCORE USING ORIGINAL (FLATTENED) FEATURES (Random Forest Classifier)",
      clf.score(X_test, y_test, sample_weight=sample_weight_vect))

# (a simple classifier) Ridge
clf = RidgeClassifier()
clf.fit(X_train_haar, y_train)
print("CLASS-BALANCED SCORE USING HAAR FEATURES (Ridge Classifier)",
      clf.score(X_test_haar, y_test, sample_weight=sample_weight_vect))
clf.fit(X_train, y_train)
print("CLASS-BALANCED SCORE USING ORIGINAL (FLATTENED) FEATURES (Ridge Classifier)",
      clf.score(X_test, y_test, sample_weight=sample_weight_vect))
