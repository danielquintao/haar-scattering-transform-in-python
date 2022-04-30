import numpy as np
import matplotlib.pyplot as plt


def read_ECG5000(train=True):
    path = "../data/ECG5000_TRAIN.txt" if train else "../data/ECG5000_TEST.txt"
    data = np.loadtxt(path)
    X, y = data[:, 1:], data[:, 0]
    return X, y


if __name__ == '__main__':
    X_train, y_train = read_ECG5000(train=True)
    plt.figure()
    plt.plot(X_train[0])
    plt.title("Example of ECG signal")
    plt.show()
    print("Class labels:", np.unique(y_train))
