from torch import nn
from torch.nn import init, Parameter
import torch

from graph_layers import KStepHGCN


class Network(torch.nn.Module):
    def __init__(self, num_output_dim, rnn_feat, input_dim, num_rnn_layers, dropout_prob, ha_len,
                 ha_hidden_dim, ha_out_dim, enc_feature, G_list):
        super(Network, self).__init__()
        self.num_output_dim = num_output_dim
        self.rnn_feat = rnn_feat
        self.num_input_dim = input_dim
        self.num_rnn_layers = num_rnn_layers

        self.dropout_prob = dropout_prob

        self.ha_len = ha_len
        self.ha_hidden_dim = ha_hidden_dim
        self.ha_out_dim = ha_out_dim
        self.enc_feature = enc_feature

        self.num_hidden_fc_dim = self.enc_feature // 2

        self.encoder = KStepHGCN(in_channels=self.num_input_dim,
                                 hidden_channels=self.num_hidden_fc_dim,
                                 out_channels=self.enc_feature,
                                 G_list=G_list,
                                 has_bias=False, dropout_prob=self.dropout_prob
                                 )

        self.encoder_hidden_fc_layer = nn.Linear(self.enc_feature, self.num_hidden_fc_dim)
        self.encoder_output_layer = nn.Linear(self.num_hidden_fc_dim, self.num_output_dim)

        self.in_feature = 2901
        self.lstm = nn.LSTM(self.in_feature, self.rnn_feat, num_layers=self.num_rnn_layers)

        self.hidden_fc_layer = nn.Linear(self.rnn_feat, self.rnn_feat * 2)
        self.output_layer = nn.Linear(self.rnn_feat * 2, self.in_feature)

        self.ha_encoder = KStepHGCN(in_channels=self.ha_len,
                                    hidden_channels=self.ha_hidden_dim,
                                    out_channels=self.ha_out_dim,
                                    G_list=G_list,
                                    has_bias=False, dropout_prob=self.dropout_prob
                                    )
        self.merge_layer = MergeLayer()

    def forward(self, sequences_x, ha):
        encoder_outputs, score_list = self.encoder(sequences_x)
        x = self.encoder_hidden_fc_layer(encoder_outputs)
        x = self.encoder_output_layer(x)
        output, [hidden, cell] = self.lstm(x[:, :, :, 0])

        decoder_hidden = self.hidden_fc_layer(hidden[-1])
        out = self.output_layer(decoder_hidden)
        out = out.unsqueeze(0)
        out = out.unsqueeze(-1)

        ha = torch.cat([ha[i] for i in range(self.ha_len)], dim=2)
        ha = ha.unsqueeze(0)
        ha_encoding, _ = self.ha_encoder(ha)

        return self.merge_layer(out, ha_encoding), score_list


class MergeLayer(nn.Module):
    def __init__(self):
        super(MergeLayer, self).__init__()
        self.weight_1 = Parameter(torch.Tensor(2901, 1))
        self.weight_2 = Parameter(torch.Tensor(2901, 1))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight_1)
        init.ones_(self.weight_2)

    def forward(self, x, ha):
        return x * self.weight_1 + ha * self.weight_2
