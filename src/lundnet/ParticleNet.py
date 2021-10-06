# This file is part of LundNet by F. Dreyer and H. Qu

from __future__ import print_function

import dgl
from dgl.transform import remove_self_loop
from .dgl_utils import segmented_knn_graph
import torch
import torch.nn as nn
import numpy as np

from lundnet.EdgeConv import EdgeConvBlock


class ParticleNet(nn.Module):
    r"""
    DGL implementation of "ParticleNet: Jet Tagging via Particle Clouds" (https://arxiv.org/abs/1902.08570).
    """

    def __init__(self,
                 input_dims,
                 num_classes,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params=[(128, 0.1)],
                 use_fusion=False,
                 **kwargs):
        super(ParticleNet, self).__init__(**kwargs)

        self.bn_fts = nn.BatchNorm1d(input_dims)

        self.k_neighbors = []
        self.edge_convs = nn.ModuleList()
        for idx, layer_param in enumerate(conv_params):
            k, channels = layer_param
            in_feat = input_dims if idx == 0 else conv_params[idx - 1][1][-1]
            self.edge_convs.append(EdgeConvBlock(in_feat=in_feat, out_feats=channels))
            self.k_neighbors.append(k)

        self.use_fusion = use_fusion
        if self.use_fusion:
            in_chn = sum(x[-1] for _, x in conv_params)
            out_chn = np.clip((in_chn // 128) * 128, 128, 1024)
            self.fusion_block = nn.Sequential(nn.Linear(in_chn, out_chn), nn.ReLU(), nn.BatchNorm1d(out_chn))

        fcs = []
        for idx, layer_param in enumerate(fc_params):
            channels, drop_rate = layer_param
            if idx == 0:
                in_chn = out_chn if self.use_fusion else conv_params[-1][1][-1]
            else:
                in_chn = fc_params[idx - 1][0]
            fcs.append(nn.Sequential(nn.Linear(in_chn, channels), nn.ReLU(), nn.Dropout(drop_rate)))
        fcs.append(nn.Linear(fc_params[-1][0], num_classes))
        self.fc = nn.Sequential(*fcs)

    def forward(self, batch_graph, features):
        g = batch_graph
        segs = batch_graph.batch_num_nodes().cpu().numpy().tolist()
        fts = self.bn_fts(features)
        outputs = []
        for idx, (k, conv) in enumerate(zip(self.k_neighbors, self.edge_convs)):
            if idx > 0:
                g = remove_self_loop(segmented_knn_graph(fts, k + 1, segs)).to(features.device)
            fts = conv(g, fts)
            if self.use_fusion:
                outputs.append(fts)
        if self.use_fusion:
            fts = self.fusion_block(torch.cat(outputs, dim=1))

        batch_graph.ndata['fts'] = fts
        x = dgl.mean_nodes(batch_graph, 'fts')
        return self.fc(x)
