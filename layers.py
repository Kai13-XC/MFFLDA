import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#GAT代码
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903. This part of code refers to the implementation of https://github.com/Diego999/pyGAT.git
    """

    def __init__(self, in_features, out_features, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # （N，N）

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)

        h_prime = torch.matmul(attention, Wh)
        return F.relu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)   # 每行重复N次
        Wh_repeated_alternating = Wh.repeat(N, 1)    # 整个Wh重复N次
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)
    # def _prepare_attentional_mechanism_input(self, Wh, chunk_size=1000, device='cuda'):
    #     N = Wh.size(0)  # number of nodes
    #     all_combinations_matrix = []
    #
    #     for i in range(0, N, chunk_size):
    #         end_i = min(i + chunk_size, N)
    #         Wh_repeated_in_chunks = Wh[i:end_i].repeat_interleave(N, dim=0).to(device)
    #         Wh_repeated_alternating = Wh.repeat(end_i - i, 1).to(device)
    #         all_combinations_matrix.append(torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1))
    #
    #     all_combinations_matrix = torch.cat(all_combinations_matrix, dim=0)
    #     return all_combinations_matrix.view(N, N, 2 * self.out_features)

    # def _prepare_attentional_mechanism_input(self, Wh):
    #     N = Wh.size(0)  # number of nodes
    #     out_features = Wh.size(1)  # number of features per node
    #
    #     # Create a tensor of shape (N, N, out_features) for the repeated chunks
    #     Wh_repeated_in_chunks = Wh.unsqueeze(1).expand(N, N, out_features)
    #     # Create a tensor of shape (N, N, out_features) for the repeated alternating
    #     Wh_repeated_alternating = Wh.unsqueeze(0).expand(N, N, out_features)
    #
    #     # Concatenate along the last dimension to get the all_combinations_matrix of shape (N, N, 2 * out_features)
    #     all_combinations_matrix = torch.cat((Wh_repeated_in_chunks, Wh_repeated_alternating), dim=2)
    #
    #     return all_combinations_matrix.view(N, N, 2 * out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

#自注意力机制
class selfattention(nn.Module):
    def __init__(self, sample_size, d_k, d_v):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.query = nn.Linear(sample_size, d_k)
        self.key = nn.Linear(sample_size, d_k)
        self.value = nn.Linear(sample_size, d_v)

    def forward(self, x):
        x =x.T
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        att = torch.matmul(q, k.transpose(0, 1)) / np.sqrt(self.d_k)
        att = torch.softmax(att, dim=1)
        output = torch.matmul(att, v)
        return output.T

#GCN代码
class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        output = torch.matmul(adj, Wh)
        return F.relu(output)

#GAC代码
class GraphAttentionLayer_GAC(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.2):
        super(GraphAttentionLayer_GAC, self).__init__()
        self.alpha = alpha
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N*N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2*h.size(1))
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(2))
        attention = F.softmax(e, dim=1)
        h_prime = torch.matmul(attention, h)
        return F.elu(h_prime)

class GAC(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GAC, self).__init__()
        self.gc1 = GraphAttentionLayer_GAC(input_dim, hidden_dim)
        self.gc2 = GraphAttentionLayer_GAC(hidden_dim, output_dim)

    def forward(self, input, adj):
        x = F.relu(self.gc1(input, adj))
        x = self.gc2(x, adj)
        return x

class GraphSAGELayer(nn.Module):
    def __init__(self, in_features, out_features, agg_method="mean"):
        super(GraphSAGELayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.agg_method = agg_method

        # 线性层和批量归一化层
        self.linear = nn.Linear(in_features * 2, out_features, bias=False)
        self.bn = nn.BatchNorm1d(out_features)

        # 池化层
        self.pool = nn.AdaptiveAvgPool1d(1)

        # LSTM层
        self.lstm = nn.LSTM(in_features, out_features, batch_first=True)

    def forward(self, x, adj):
        if self.agg_method == "mean":
            out = torch.spmm(adj, x)
        elif self.agg_method == "sum":
            out = torch.spmm(adj, x)
        elif self.agg_method == "max":
            out = torch.spmm(adj, x)
            #out = torch.max(out, dim=1, keepdim=True)[0]
            out, _ = torch.max(out, dim=1, keepdim=True)
            #out = out.repeat(1, x.shape[1])  # 扩展维度
            out = out.expand(-1, x.shape[1])  # 让维度匹配 x
        elif self.agg_method == "pool":
            out = torch.spmm(adj, x)
            out = self.pool(out.unsqueeze(0)).squeeze(0)
        elif self.agg_method == "lstm":
            out = torch.spmm(adj, x)
            # out, _ = self.lstm(out.unsqueeze(0))
            # out = out.squeeze(0)
            out = out.unsqueeze(1)  # 增加 seq_len 维度
            out, _ = self.lstm(out)
            out = out.squeeze(1)
        else:
            raise NotImplementedError

        # 将节点自身的特征与邻居节点的特征进行拼接
        out = torch.cat([x, out], dim=1)

        # 线性变换和批量归一化
        out = self.linear(out)
        out = self.bn(out)
        return out