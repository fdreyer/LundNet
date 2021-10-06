# This file is part of LundNet by F. Dreyer and H. Qu

from __future__ import print_function

import dgl
import networkx as nx
import numpy as np
from dgl.transform import remove_self_loop
from .dgl_utils import knn_graph
from torch.utils.data import Dataset
from .JetTree import JetTree, LundCoordinates
from .read_data import Jets
import torch
import torch.nn.functional as F
try:
    from uproot_methods import TLorentzVectorArray, TLorentzVector
except ImportError:
    from uproot3_methods import TLorentzVectorArray, TLorentzVector
import time
import pandas as pd

groomer = None
dump_number_of_nodes = False


class DGLGraphDatasetLund(Dataset):

    fill_secondary = True
    node_coordinates = 'eta-phi'  # 'lund'

    def __init__(self, filepath_bkg, filepath_sig, nev=-1):
        super(DGLGraphDatasetLund, self).__init__()
        print('Start loading dataset %s (bkg) and %s (sig)' % (filepath_bkg, filepath_sig))
        tic = time.process_time()
        reader_bkg = Jets(filepath_bkg, nev, groomer=groomer)
        reader_sig = Jets(filepath_sig, nev, groomer=groomer)
        # attempt at using less memory
        self.data = []
        self.label = []
        for jet in reader_bkg:
            self.data += [self._build_tree(JetTree(jet))]
            self.label += [0]
        for jet in reader_sig:
            self.data += [self._build_tree(JetTree(jet))]
            self.label += [1]
        print(' ... Total time to read input files + construct the graphs for {num} jets: {ts} seconds'.format(
            num=len(self.label), ts=time.process_time() - tic))
        if dump_number_of_nodes:
            df = pd.DataFrame({'num_nodes': np.array(
                [g.number_of_nodes() for g in self.data]), 'label': np.array(self.label)})
            df.to_csv('num_nodes_lund_net_ktmin_%s_deltamin_%s.csv' % (JetTree.ktmin, JetTree.deltamin))
        self.label = torch.tensor(self.label, dtype=torch.float32)

    def _build_tree(self, root):
        g = nx.Graph()
        jet_p4 = TLorentzVector(*root.node)

        def _rec_build(nid, node):
            branches = [node.harder, node.softer] if DGLGraphDatasetLund.fill_secondary else [node.harder]
            for branch in branches:
                if branch is None or branch.lundCoord is None:
                    # stop when reaching the leaf nodes
                    # we do not add the leaf nodes to the graph/tree as they do not have Lund coordinates
                    continue
                cid = g.number_of_nodes()
                if DGLGraphDatasetLund.node_coordinates == 'lund':
                    spatialCoord = branch.lundCoord.state()[:2]
                else:
                    node_p4 = TLorentzVector(*branch.node)
                    spatialCoord = np.array(
                        [delta_eta_reflect(node_p4, jet_p4),
                         node_p4.delta_phi(jet_p4)],
                        dtype='float32')
                g.add_node(cid, coordinates=spatialCoord, features=branch.lundCoord.state())
                g.add_edge(cid, nid)
                _rec_build(cid, branch)
        # add root
        if root.lundCoord is not None:
            if DGLGraphDatasetLund.node_coordinates == 'lund':
                spatialCoord = root.lundCoord.state()[:2]
            else:
                spatialCoord = np.zeros(2, dtype='float32')
            g.add_node(0, coordinates=spatialCoord, features=root.lundCoord.state())
            _rec_build(0, root)
        else:
            # when a jet has only one particle (?)
            g.add_node(0, coordinates=np.zeros(2, dtype='float32'),
                       features=np.zeros(LundCoordinates.dimension, dtype='float32'))
        ret = dgl.from_networkx(g, node_attrs=['coordinates', 'features'])
        # print(ret.number_of_nodes())
        return ret

    @property
    def num_features(self):
        return self.data[0].ndata['features'].shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.data[i]
        y = self.label[i]
        return x, y


class DGLGraphDatasetParticle(Dataset):

    def __init__(self, filepath_bkg, filepath_sig, nev=-1):
        super(DGLGraphDatasetParticle, self).__init__()
        print('Start loading dataset %s (bkg) and %s (sig)' % (filepath_bkg, filepath_sig))
        tic = time.process_time()
        reader_bkg = Jets(filepath_bkg, nev, pseudojets=False, groomer=groomer)
        reader_sig = Jets(filepath_sig, nev, pseudojets=False, groomer=groomer)
        # Format of jets_bkg/jets_sig:
        # [# jet
        #     [ # particle
        #         [px, py, pz, E],
        #       # particle 2
        #         [px, py, pze, E]
        #     ],
        #  # jet 2
        #     #.....
        # ]
        # attempt at saving memory use:
        self.data = []
        self.label = []
        for constits in reader_bkg:
            self.data += [self._build_graph(constits)]
            self.label += [0]
        for constits in reader_sig:
            self.data += [self._build_graph(constits)]
            self.label += [1]
        print(' ... Total time to read input files + construct the graphs for {num} jets: {ts} seconds'.format(
            num=len(self.label), ts=time.process_time() - tic))
        if dump_number_of_nodes:
            df = pd.DataFrame({'num_nodes': np.array(
                [g.number_of_nodes() for g in self.data]), 'label': np.array(self.label)})
            df.to_csv('num_nodes_particle_net.csv')
        self.label = torch.tensor(self.label, dtype=torch.float32)

    def _build_graph(self, constits):
        constits_p4 = TLorentzVectorArray.from_cartesian(*list(zip(*constits)))
        jet_p4 = constits_p4.sum()
        spatialCoord = np.stack([delta_eta_reflect(constits_p4, jet_p4), constits_p4.delta_phi(jet_p4)], axis=1)
        energyFeatures = np.log(np.stack([constits_p4.pt, constits_p4.energy], axis=1))
        features = np.concatenate([spatialCoord, energyFeatures], axis=1)
        ret = dgl.DGLGraph()
        ret.add_nodes(
            len(constits),
            {'coordinates': torch.tensor(spatialCoord, dtype=torch.float32),
             'features': torch.tensor(features, dtype=torch.float32)})
        # print(ret.number_of_nodes())
        return ret

    @property
    def num_features(self):
        return self.data[0].ndata['features'].shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.data[i]
        y = self.label[i]
        return x, y


def delta_eta_reflect(constits_p4, jet_p4):
    deta = constits_p4.eta - jet_p4.eta
    return deta if jet_p4.eta > 0 else -deta


def pad_array(a, min_len=20, pad_value=0):
    if a.shape[0] < min_len:
        return F.pad(a, (0, 0, 0, min_len - a.shape[0]), mode='constant', value=pad_value)
    else:
        return a


class _SimpleCustomBatch:

    def __init__(self, data, k, min_nodes=20):
        transposed_data = list(zip(*data))
        graphs = []
        features = []
        for g in transposed_data[0]:
            nng = remove_self_loop(knn_graph(g.ndata['coordinates'], min(g.number_of_nodes(), k + 1)))
            if nng.number_of_nodes() < min_nodes:
                nng.add_nodes(min_nodes - nng.number_of_nodes())
            graphs.append(nng)
            fts = pad_array(g.ndata['features'], min_nodes, 0)
            features.append(fts)
            assert(nng.number_of_nodes() == fts.shape[0])
        self.batch_graph = dgl.batch(graphs)
        self.features = torch.cat(features, 0)
        self.label = torch.tensor(transposed_data[1])

    def pin_memory(self):
        self.features = self.features.pin_memory()
        self.label = self.label.pin_memory()
        return self


def collate_wrapper(batch, k):
    return _SimpleCustomBatch(batch, k)


class _LundTreeBatch:

    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.batch_graph = dgl.batch(transposed_data[0])
        self.batch_graph.ndata.pop('coordinates')  # drop (eta, phi) coordinates
        self.features = self.batch_graph.ndata.pop('features')
        self.label = torch.tensor(transposed_data[1])

    def pin_memory(self):
        self.features = self.features.pin_memory()
        self.label = self.label.pin_memory()
        return self


def collate_wrapper_tree(batch):
    return _LundTreeBatch(batch)
