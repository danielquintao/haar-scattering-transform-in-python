# Haar Scattering Transform in Python
A non-official implementation of "Unsupervised Deep Haar Scattering on Graphs, Chen X., Cheng X., Mallat S. 2014"

## Data

Please download [MNIST](http://yann.lecun.com/exdb/mnist/) and paste it inside ``./data/``.

## Code overview

All code files are in ```./src/```.

Files with "utilities":
- ``haar_scattering_transform.py`` has an implementation of a class that ... computes the Haar Scattering Transform !
It is supposed to be clear from the docstrings, but in summary: we instantiate it with an ``igraph`` object representing
the structure of the domain where the signal lives (so that the pairings to pass from layer j to layer j+1 are computed
during initialization), and it has a method ``get_haar_scattering_transform`` that receives a (flattened) signal living
in that (graph) domain and computes its scalar or boolean Haar Scattering Transform, depending on the dtype of the input
(this is automatically detected).
- ``matching.py`` calls [blossalg](https://github.com/nenb/blossalg) methods that help us matching "nodes" in the Haar 
Scattering Transform. The package ``blossalg`` goal is to implement the Blossom Algorithm and it is still very young, 
but it works for this project. Install it at your own risk ! 
- ``images2graphs.py`` has some utilities to build the domain graph for image grids and convert images to the right format.
- ``read_MNIST.py`` reads the MNIST that we should be able to find under ``./data/``.

Illustration files: all the files starting with ``example_[...]``