# This file is part of LundNet by F. Dreyer and H. Qu

from __future__ import print_function

import dgl.function as fn
import torch.nn as nn


class EdgeConvBlock(nn.Module):
    r"""EdgeConv layer.
    Introduced in "Dynamic Graph CNN for Learning on Point Clouds" (https://arxiv.org/pdf/1801.07829).
    Code adapted from https://github.com/dmlc/dgl/blob/master/python/dgl/nn/pytorch/conv/edgeconv.py.
    """

    def __init__(self, in_feat, out_feats, batch_norm=True, activation=True):
        super(EdgeConvBlock, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.num_layers = len(out_feats)

        out_feat = out_feats[0]
        self.theta = nn.Linear(in_feat, out_feat, bias=False if self.batch_norm else True)
        self.phi = nn.Linear(in_feat, out_feat, bias=False if self.batch_norm else True)
        self.fcs = nn.ModuleList()
        for i in range(1, self.num_layers):
            self.fcs.append(nn.Linear(out_feats[i - 1], out_feats[i], bias=False if self.batch_norm else True))

        if batch_norm:
            self.bns = nn.ModuleList()
            for i in range(self.num_layers):
                self.bns.append(nn.BatchNorm1d(out_feats[i]))

        if activation:
            self.acts = nn.ModuleList()
            for i in range(self.num_layers):
                self.acts.append(nn.ReLU())

        if in_feat == out_feats[-1]:
            self.sc = None
        else:
            self.sc = nn.Linear(in_feat, out_feats[-1], bias=False if self.batch_norm else True)
            self.sc_bn = nn.BatchNorm1d(out_feats[-1])

        if activation:
            self.sc_act = nn.ReLU()

    def message(self, edges):
        theta_x = self.theta(edges.dst['x'] - edges.src['x'])
        phi_x = self.phi(edges.src['x'])
        return {'e': theta_x + phi_x}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['x'] = h
            # generate the message and store it on the edges
            g.apply_edges(self.message)
            # process the message
            e = g.edata['e']
            for i in range(self.num_layers):
                if i > 0:
                    e = self.fcs[i - 1](e)
                if self.batch_norm:
                    e = self.bns[i](e)
                if self.activation:
                    e = self.acts[i](e)
            g.edata['e'] = e
            # pass the message and update the nodes
            g.update_all(fn.copy_e('e', 'e'), fn.mean('e', 'x'))
            # shortcut connection
            x = g.ndata.pop('x')
            g.edata.pop('e')
            if self.sc is None:
                sc = h
            else:
                sc = self.sc(h)
                if self.batch_norm:
                    sc = self.sc_bn(sc)
            if self.activation:
                return self.sc_act(x + sc)
            else:
                return x + sc
