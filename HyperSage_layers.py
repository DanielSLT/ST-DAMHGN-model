from torch import nn


class HyperSage(nn.Module):
    def __init__(self, dim_in, dim_out, matrix_dict, has_bias, dropout_prob):
        super(HyperSage, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.edge_sampled = matrix_dict["edge_sampled"]
        self.H = matrix_dict["H"]
        self.H_sampled_coef = matrix_dict["H_sampled_coef"]
        self.fc = nn.Linear(self.dim_in, self.dim_out, bias=has_bias)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.activation = nn.ReLU()

    def forward(self, feats):
        num_edge_sampled = self.edge_sampled.sum(1, keepdim=True)
        x = self.edge_sampled.matmul(feats) / num_edge_sampled

        num_edge = self.H.sum(1, keepdim=True)
        scaled_H = self.H * self.H_sampled_coef
        x = scaled_H.matmul(x) / num_edge

        x = feats + x
        x = self.fc(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
