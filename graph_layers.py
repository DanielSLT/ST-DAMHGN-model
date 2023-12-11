from torch import nn
import torch

from DHGNN_layers import DHGLayer


class KStepHGCN(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            out_channels,
            G_list,
            has_bias,
            dropout_prob

    ):
        super(KStepHGCN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.G_list = G_list
        layer_list = [
            DHGLayer(in_channels,
                     hidden_channels, hidden_channels, G_list,
                     has_bias, dropout_prob),
            DHGLayer(hidden_channels, hidden_channels,
                     out_channels, G_list,
                     has_bias, dropout_prob)]

        self.hgcn_layers = nn.ModuleList(layer_list)

    def forward(self, x):
        score_list = []
        for layer in self.hgcn_layers:
            x, scores = layer(x)
            score_list.append(scores)
            x = torch.relu(x)
        return x, score_list
