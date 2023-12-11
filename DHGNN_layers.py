import torch
from torch import nn

from HyperSage_layers import HyperSage


class Attention(nn.Module):
    def __init__(self, dim_ft, hidden):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim_ft, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        self.h = nn.Parameter(torch.empty(size=(1, 1, 1, 1, dim_ft)))
        nn.init.xavier_uniform_(self.h.data, gain=1.414)
        self.W = nn.Parameter(torch.empty(size=(dim_ft, hidden)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * hidden, 1)))  # concat(V,NeigV)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()

    def forward(self, ft):
        Wf = ft.matmul(self.W)
        Wh = self.h.matmul(self.W)
        Wh = Wh.repeat(ft.shape[0], ft.shape[1], ft.shape[2], ft.shape[3], 1)
        e = torch.cat([Wf, Wh], dim=4)
        e = e.matmul(self.a)
        e = self.leakyrelu(e)
        scores = torch.softmax(e, 2)
        return self.relu((scores * ft).sum(2)), scores


class DHGLayer(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, G_list, has_bias=True, drop_out_rate=0):
        super().__init__()

        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.fc = nn.Linear(self.dim_hidden, self.dim_out, bias=has_bias)
        self.dropout = nn.Dropout(p=drop_out_rate)
        self.activation = nn.ReLU()
        self.hgc_s = HyperSage(self.dim_in, self.dim_hidden, G_list[0], has_bias, drop_out_rate)
        self.hgc_c = HyperSage(self.dim_in, self.dim_hidden, G_list[1], has_bias, drop_out_rate)
        self.hgc_poi_s = HyperSage(self.dim_in, self.dim_hidden, G_list[2], has_bias, drop_out_rate)
        self.hgc_poi_c = HyperSage(self.dim_in, self.dim_hidden, G_list[3], has_bias, drop_out_rate)
        self.att = Attention(self.dim_hidden, hidden=max(self.dim_hidden // 4, 4))

    def _graph_conv(self, func, x):
        return func(x)

    def _fc(self, x):
        return self.activation(self.fc(self.dropout(x)))

    def forward(self, feats):
        hyperedges = []
        xc = self._graph_conv(self.hgc_c, feats)
        hyperedges.append(xc)
        xs = self._graph_conv(self.hgc_s, feats)
        hyperedges.append(xs)
        x_poi_c = self._graph_conv(self.hgc_poi_c, feats)
        hyperedges.append(x_poi_c)
        x_poi_s = self._graph_conv(self.hgc_poi_s, feats)
        hyperedges.append(x_poi_s)

        x = torch.stack(hyperedges, dim=2)
        x, scores = self.att(x)
        x = self._fc(x)
        return x, scores
