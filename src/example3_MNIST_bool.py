"""Visualizing boolean Haar Scattering Transforms of binarized MNIST (for the scalar case, run read_MNIST.py as script)

"""

import numpy as np
import matplotlib.pyplot as plt
from haar_scattering_transform import HaarScatteringTransform
from unstructured2graphs import graph_for_grid, project_signal, signal2image
from read_MNIST import read_10000_from_MNIST


########################################################################################################################
# HAAR SCATTERING TRANSFORM FOR BOOLEAN VARIABLES ON BINARIZED MNIST
########################################################################################################################

X_train, y_train = read_10000_from_MNIST(binarize=True)

im = X_train[0]
J = 3

fig = plt.figure(constrained_layout=True, figsize=(10, 4))
ax = fig.subplots(J + 1, 2 ** J)

ax[0][0].imshow(im, cmap="gray")
ax[0][0].set_xticks([])
ax[0][0].set_yticks([])
ax[0][0].set_ylabel("j=0")

haar = HaarScatteringTransform(graph_for_grid(im.shape), J=J)
transform = haar.get_haar_scattering_transform(project_signal(im))
for j, coeffs in enumerate(transform[1:], start=1):
    N, Q = coeffs.shape
    imgs = [np.zeros_like(project_signal(im)) for _ in range(Q)]
    for n in range(N):
        v_set = haar.get_receptive_field(j, n)
        for q in range(Q):
            imgs[q][np.array(list(v_set))] = coeffs[n, q]
    for q in range(len(imgs)):
        ax[j][q].imshow(signal2image(imgs[q], im.shape), cmap="gray")
        ax[j][q].set_xticks([])
        ax[j][q].set_yticks([])
    ax[j][0].set_ylabel("j=" + str(j))

for q in range(2 ** J):
    ax[J][q].set_xlabel("q=" + str(q))

for j in range(J + 1):
    for k in range(2 ** j, 2 ** J):
        ax[j][k].axis('off')
        ax[j][k].grid(False)
        ax[j][k].set_xticks([])
        ax[j][k].set_yticks([])

plt.show()



