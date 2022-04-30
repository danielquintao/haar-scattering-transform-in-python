import igraph
import numpy as np


########################################################################################################################
# graph_for_grid and project_signal are very dumb but they communicate (i.e. they were made to be used together)
########################################################################################################################


# IMAGES


def graph_for_grid(shape):
    """ create igraph Graph representing the gris of an image

    :param shape: tuple, list or array-like with 2 elements (shape of the images)
    :return: igraph
    """
    if len(shape) != 2:
        raise ValueError("wrong shape; should be 2D")
    return igraph.Graph.Lattice([*shape], circular=False)


def project_signal(img):
    """return signal in right format to be fed into a HaarScatteringTransform object initiated for images of that shape

    :param img: numpy array with 2 dimensions
    :return: 1D numpy array
    """
    if not isinstance(img, np.ndarray) or len(img.shape) != 2:
        raise ValueError("wrong format; img should be 2D numpy array")
    return img.flatten()


def signal2image(signal, shape):
    """reverse project signal

    :param signal: 1D array with shape[0] * shape[1]
    :param shape: shape of the original image used to build the signal
    :return: 2d numpy array
    """
    if len(shape) != 2:
        raise ValueError("wrong shape; should be 2D")
    if not isinstance(signal, np.ndarray) or signal.shape != (np.product(shape),):
        raise ValueError("wrong format; signal should be 1D numpy array with shape[0] * shape[1] entries")
    return signal.reshape(*shape)


# PERIODIC TIME SERIES


def graph_for_per_time_series(length):
    """ create igraph Graph representing the cyclic graph of a periodic time series

    :param length: int (number of samples for the time series signals)
    :return: igraph
    """
    return igraph.Graph.Ring(length)


if __name__ == "__main__":
    im = np.arange(12).reshape(4, 3)
    print(graph_for_grid(im.shape))
    print(project_signal(im))
    print(signal2image(project_signal(im), im.shape))
