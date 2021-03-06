""" This is just a visualization of the Haar scattering Transform applied on an image

Due to technical issues with the Blossom Algorithm, we need to subsample the example image to avoid
memory issues. We can still recognize the scikit-image Astronaut example:
https://scikit-image.org/docs/dev/api/skimage.data.html#skimage.data.astronaut

"""

import numpy as np
from skimage import data, color
from skimage.transform import rescale
import matplotlib.pyplot as plt
from haar_scattering_transform import HaarScatteringTransform
from unstructured2graphs import graph_for_grid, project_signal, signal2image

im = color.rgb2gray(data.astronaut())
im = rescale(im, 0.0625)  # ATTENTION: Python crashes during blossom algorithm if image is too large
print(im.shape)

plt.figure()
plt.imshow(im, cmap="gray")
plt.show()

haar = HaarScatteringTransform(graph_for_grid(im.shape), J=2)
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



