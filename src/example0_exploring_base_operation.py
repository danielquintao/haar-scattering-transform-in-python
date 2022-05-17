"""Visualizing the type of information represented by sums and absolute differences in Haar scattering transf. algorithm

"""

import numpy as np
import matplotlib.pyplot as plt
from haar_scattering_transform import HaarScatteringTransform
from unstructured2graphs import graph_for_grid, project_signal, signal2image
import pprint


im0 = np.zeros((8, 8)).astype(float)  # black
im1 = np.ones((8, 8)).astype(float)  # white
im2 = np.tile(np.eye(2), (4, 4)).astype(float)  # checker board
im3 = np.tile(np.diag([1, 0, 0, 0]), (2, 2)).astype(float)  # just visualize it
im4 = 1 - im2  # checkerboard starting from black tile

J = 2

for im in [im0, im1, im2, im3, im4]:
    fig = plt.figure(constrained_layout=True, figsize=(6, 4))
    ax = fig.subplots(J + 1, 2 ** J)

    ax[0][0].imshow(im, cmap="gray", vmin=0, vmax=1)
    ax[0][0].set_xticks([])
    ax[0][0].set_yticks([])
    ax[0][0].set_ylabel("j=0")

    haar = HaarScatteringTransform(graph_for_grid(im.shape), J=J)
    transform = haar.get_haar_scattering_transform(project_signal(im))

    pprint.pp(haar.multi_resolution_approx_pairings)

    for j, coeffs in enumerate(transform[1:], start=1):
        N, Q = coeffs.shape
        imgs = [np.zeros_like(project_signal(im)) for _ in range(Q)]
        for n in range(N):
            v_set = haar.get_receptive_field(j, n)
            for q in range(Q):
                imgs[q][np.array(list(v_set))] = coeffs[n, q]
        for q in range(len(imgs)):
            ax[j][q].imshow(signal2image(imgs[q], im.shape), cmap="gray", vmin=0, vmax=1)
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

# visualize matching of the last layer
for j, coeffs in enumerate(transform[1:], start=1):
    multires_map = np.zeros(64)
    # coeffs = transform[-1]
    N, Q = coeffs.shape
    for n in range(N):
        v_set = haar.get_receptive_field(j, n)  # J -> j
        multires_map[np.array(list(v_set))] = n
    plt.figure()
    mapping = signal2image(multires_map, (8, 8))
    plt.imshow(mapping, cmap='tab20')
    for (j, i), label in np.ndenumerate(mapping):
        plt.text(i, j, str(int(label)), ha='center', va='center')
    plt.xticks([])
    plt.yticks([])
    plt.show()
