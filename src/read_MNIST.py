import numpy as np
import matplotlib.pyplot as plt
from haar_scattering_transform import HaarScatteringTransform
from images2graphs import graph_for_grid, project_signal, signal2image
import gzip
import os


def read_10000_from_MNIST(folder='../data/', train=True, one_hot=False, binarize=False):
    """reads MNIST, adapted from https://stackoverflow.com/a/53570674

    """
    # IMAGES:
    if train:
        f = gzip.open(os.path.join(folder, 'train-images-idx3-ubyte.gz'), 'r')
    else:
        f = gzip.open(os.path.join(folder, 't10k-images-idx3-ubyte.gz'), 'r')

    image_size = 28
    num_images = 10000

    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size, image_size)
    data /= 255.  # rescale to gray scale

    if binarize:
        data = data > 0.5

    # LABELS:
    if train:
        f = gzip.open(os.path.join(folder, 'train-labels-idx1-ubyte.gz'), 'r')
    else:
        f = gzip.open(os.path.join(folder, 't10k-labels-idx1-ubyte.gz'), 'r')
    f.read(8)
    buf = f.read(num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    # convert to one_hot
    if one_hot:
        labels_oh = np.zeros((num_images, 10))
        for i, digit in enumerate(labels):
            labels_oh[i, digit] = 1
        labels = labels_oh

    return data, labels


if __name__ == "__main__":
    X_train, y_train = read_10000_from_MNIST()
    print("Shape of X, y:", X_train.shape, y_train.shape)

    ####################################################################################################################
    # VISUALIZE HAR SCATTERING TRANSFORM ON A ELEMENT FROM MNIST
    ####################################################################################################################

    im = X_train[0]

    plt.figure()
    plt.imshow(im, cmap="gray")
    plt.show()

    haar = HaarScatteringTransform(graph_for_grid(im.shape), J=3)
    transform = haar.get_haar_scattering_transform(project_signal(im))
    for j, coeffs in enumerate(transform[1:], start=1):
        N, Q = coeffs.shape
        imgs = [np.zeros_like(project_signal(im)) for _ in range(Q)]
        for n in range(N):
            v_set = haar.get_receptive_field(j, n)
            for q in range(Q):
                imgs[q][np.array(list(v_set))] = coeffs[n, q]
        fig, ax = plt.subplots(1, Q)
        for q in range(len(imgs)):
            ax[q].imshow(signal2image(imgs[q], im.shape), cmap="gray")
        plt.show()



