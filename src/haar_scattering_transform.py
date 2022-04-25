import numpy as np
import igraph
import copy
from matching import compute_max_matching
from collections import defaultdict
import pprint


class HaarScatteringTransform:
    def __init__(self, graph_domain: igraph.Graph, J: int = 2):
        """ initialize object for Haar scattering Transform of incoming signals in a predefined graph structure

        NOTE: in this implementation, coefficients in layer j can be paired ONLY IF their corresponding vertex sets
        are connected

        :param graph_domain: igraph.Graph instance
        :param J: desired maximum scale/ may be automatically downgraded
        """
        self.graph_domain = copy.deepcopy(graph_domain)
        self.N = graph_domain.vcount()
        self.J = min(J, np.floor(np.log2(self.N)))
        g = copy.deepcopy(graph_domain)  # we will modify it --> deepcopy
        # MULTIRESOLUTION PAIRING MAPS
        self.multi_resolution_approx_pairings = []

        def multi_resolution_map(g, j):
            adjacency_list = g.get_adjacency().data  # list of lists
            l = [[idx for idx, v in enumerate(row) if v] for row in adjacency_list]
            matching = compute_max_matching(l)
            matching_set = set()
            for k, v in matching.items():
                matching_set.add((min(k, v), max(k, v)))
            if len(matching_set) < g.vcount() / 2:
                print("{} unmatched vertices in layer {}".format(g.vcount() - 2 * len(matching_set), j))
            for i, (u, v) in enumerate(matching_set):
                g.vs(u)["pair"] = i
                g.vs(v)["pair"] = i
            # unmatched
            for v in g.vs:
                if v["pair"] is None:
                    i += 1
                    v["pair"] = i
            clus = igraph.VertexClustering.FromAttribute(g, attribute="pair", intervals=list(range(i + 1)))
            membership = [v["pair"] for v in g.vs]
            is_paired_list = [v["pair"] < len(matching_set) for v in g.vs]
            return membership, clus.cluster_graph(), is_paired_list  # membership is a list of pair index e.g. [0,1,0,1]

        for j in range(J):
            membership, g, is_paired_list = multi_resolution_map(g, j)
            # convert to new_pos : (a_n, b_n) form
            membership_ = defaultdict(list)
            for pos, (new_pos, is_paired) in enumerate(zip(membership, is_paired_list)):
                if is_paired:  # ignore unpaired nodes
                    membership_[new_pos].append(pos)
            self.multi_resolution_approx_pairings.append(membership_)

    def get_haar_scattering_transform(self, signal: np.ndarray, save_memory=False):
        """computes each layer's Haar scattering transforms for a signal defined on the graph used in __init__

        :param signal: 1D numpy array with scalar signals COHERENTLY WITH THE ORDER OF THE (DOMAIN) GRAPH VERTICES
        :param save_memory: bool, whether to drop scattering coefficients when not enough paired nodes (True), or to
                            always return arrays of known size and leave 0 in non-computed coefficients. Chose False to
                            have more control on the returned shape.
        :return: list with the Haar Scattering transforms at each scale (use last layer's coefficients for classif.)
        """
        if not isinstance(signal, np.ndarray) or signal.shape not in [(len(signal),), (len(signal), 1)]:
            raise ValueError("signal in unexpected format; should be 1D numpy array or column vector (numpy array)")
        signal = signal.reshape(-1, 1)
        if self.N != len(signal):
            raise ValueError("signal of incorrect length")
        J = len(self.multi_resolution_approx_pairings)
        transform = [signal] + [
            np.zeros((len(self.multi_resolution_approx_pairings[j-1]), 2 ** j)) if save_memory
            else np.zeros((self.N // (2 ** j), 2 ** j))
            for j in range(1, J + 1)
        ]
        for j in range(1, len(transform)):
            _, n_cols = transform[j-1].shape
            for q in range(n_cols):
                pairing = self.multi_resolution_approx_pairings[j-1]
                for n, pair in pairing.items():
                    transform[j][n, 2 * q] = transform[j-1][pair[0], q] + transform[j-1][pair[1], q]
                    transform[j][n, 2 * q + 1] = abs(transform[j-1][pair[0], q] - transform[j-1][pair[1], q])
        return transform

if __name__ == "__main__":
    print(igraph.__version__)
    graph = igraph.Graph([(0, 1), (2, 3), (3, 4), (4, 5)])
    haar = HaarScatteringTransform(graph)
    pprint.pp(haar.multi_resolution_approx_pairings)
    signal = np.array([1, 10, 100, 1000, 10000, 100000])
    pprint.pp(haar.get_haar_scattering_transform(signal))

