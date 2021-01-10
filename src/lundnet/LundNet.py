# This file is part of LundNet by F. Dreyer and H. Qu

from __future__ import print_function

import dgl
import torch
import torch.nn as nn
import numpy as np

from lundnet.EdgeConv import EdgeConvBlock


class LundNet(nn.Module):

    def __init__(self,
                 input_dims,
                 num_classes,
                 conv_params=[[32, 32], [32, 32], [64, 64], [64, 64], [128, 128], [128, 128]],
                 fc_params=[(128, 0.1)],
                 use_fusion=True,
                 **kwargs):
        super(LundNet, self).__init__(**kwargs)

        self.bn_fts = nn.BatchNorm1d(input_dims)

        self.edge_convs = nn.ModuleList()
        for idx, channels in enumerate(conv_params):
            in_feat = input_dims if idx == 0 else conv_params[idx - 1][-1]
            self.edge_convs.append(EdgeConvBlock(in_feat=in_feat, out_feats=channels))

        self.use_fusion = use_fusion
        if self.use_fusion:
            in_chn = sum(x[-1] for x in conv_params)
            out_chn = np.clip((in_chn // 128) * 128, 128, 1024)
            self.fusion_block = nn.Sequential(nn.Linear(in_chn, out_chn), nn.ReLU(), nn.BatchNorm1d(out_chn))

        fcs = []
        for idx, layer_param in enumerate(fc_params):
            channels, drop_rate = layer_param
            if idx == 0:
                in_chn = out_chn if self.use_fusion else conv_params[-1][-1]
            else:
                in_chn = fc_params[idx - 1][0]
            fcs.append(nn.Sequential(nn.Linear(in_chn, channels), nn.ReLU(), nn.Dropout(drop_rate)))
        fcs.append(nn.Linear(fc_params[-1][0], num_classes))
        self.fc = nn.Sequential(*fcs)

    def forward(self, batch_graph, features):
        fts = self.bn_fts(features)
        outputs = []
        for idx, conv in enumerate(self.edge_convs):
            fts = conv(batch_graph, fts)
            if self.use_fusion:
                outputs.append(fts)
        if self.use_fusion:
            fts = self.fusion_block(torch.cat(outputs, dim=1))

        batch_graph.ndata['fts'] = fts
        x = dgl.mean_nodes(batch_graph, 'fts')
        return self.fc(x)
