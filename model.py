
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *
import csv


class MultiDeep(nn.Module):
    def __init__(self, nrnai, nrnaj, nrnaifeat, nrnajfeat, nhid, nheads, alpha):
        """Dense version of GAT."""
        super(MultiDeep, self).__init__()


        self.protein_attentions1 = [GraphAttentionLayer(nrnaifeat, nhid, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.protein_attentions1):
            self.add_module('Attention_Protein1_{}'.format(i), attention)

        self.protein_attentions2 = [GraphAttentionLayer(nhid * nheads, nhid, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.protein_attentions2):
            self.add_module('Attention_Protein2_{}'.format(i), attention)


        self.protein_prolayer1 = nn.Linear(nhid * nheads, nhid * nheads, bias=False)
        self.protein_LNlayer1 = nn.LayerNorm(nhid * nheads)

        self.protein_sage_1 = GraphSAGELayer(nhid*nheads, nhid*nheads, agg_method="mean")
        self.protein_sage_2 = GraphSAGELayer(nhid*nheads, nhid*nheads, agg_method="mean")

        self.protein_prolayer2 = nn.Linear(nhid * nheads, nhid * nheads, bias=False)
        self.protein_LNlayer2 = nn.LayerNorm(nhid * nheads)

        self.drug_attentions1 = [GraphAttentionLayer(nrnajfeat, nhid, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.drug_attentions1):
            self.add_module('Attention_Drug1_{}'.format(i), attention)

        self.drug_attentions2 = [GraphAttentionLayer(nhid * nheads, nhid, alpha=alpha, concat=True) for _ in
                                 range(nheads)]
        for i, attention in enumerate(self.drug_attentions2):
            self.add_module('Attention_Drug2_{}'.format(i), attention)
        self.drug_prolayer1 = nn.Linear(nhid * nheads, nhid * nheads, bias=False)
        self.drug_LNlayer1 = nn.LayerNorm(nhid * nheads)


        self.drug_sage_1 = GraphSAGELayer(nhid*nheads, nhid*nheads, agg_method="mean")
        self.drug_sage_2 = GraphSAGELayer(nhid*nheads, nhid*nheads, agg_method="mean")
        self.drug_prolayer2 = nn.Linear(nhid * nheads, nhid * nheads, bias=False)
        self.drug_LNlayer2 = nn.LayerNorm(nhid * nheads)

        self.FClayer1 = nn.Linear(nhid * nheads * 2, nhid * nheads * 2)
        self.FClayer2 = nn.Linear(nhid * nheads * 2, nhid * nheads * 2)
        self.FClayer3 = nn.Linear(nhid * nheads * 2, 1)
        self.output = nn.Sigmoid()

    def forward(self, protein_features, protein_adj, drug_features, drug_adj, idx_protein_drug, device): #device是用CPU还是GPU
        proteinx = torch.cat([att(protein_features, protein_adj) for att in self.protein_attentions1], dim=1) #对于每一格图注意力层 调用layer中的对用的forward方法
        proteinx = F.relu(proteinx)
        proteinx = torch.cat([att(proteinx, protein_adj) for att in self.protein_attentions2], dim=1)
        proteinx = F.relu(proteinx)
        proteinx = self.protein_prolayer1(proteinx)
        proteinx = self.protein_LNlayer1(proteinx)

        proteinx = self.protein_sage_1(proteinx, protein_adj)
        proteinx = F.relu(proteinx)
        proteinx = self.protein_sage_2(proteinx, protein_adj)
        proteinx = F.relu(proteinx)
        proteinx = self.protein_prolayer2(proteinx)
        proteinx = self.protein_LNlayer2(proteinx)

        drugx = torch.cat([att(drug_features, drug_adj) for att in self.drug_attentions1], dim=1)
        drugx = F.relu(drugx)
        drugx = torch.cat([att(drugx, drug_adj) for att in self.drug_attentions2], dim=1)
        drugx = F.relu(drugx)
        drugx = self.drug_prolayer1(drugx)
        drugx = self.drug_LNlayer1(drugx)

        drugx = self.drug_sage_1(drugx, drug_adj)
        drugx = F.relu(drugx)
        drugx = self.drug_sage_2(drugx, drug_adj)
        drugx = F.relu(drugx)

        drugx = self.drug_prolayer2(drugx)
        drugx = self.drug_LNlayer2(drugx)

        protein_drug_x = torch.cat((proteinx[idx_protein_drug[:, 0]], drugx[idx_protein_drug[:, 1]]), dim=1)
        protein_drug_x = protein_drug_x.to(device)
        protein_drug_x = self.FClayer1(protein_drug_x)
        protein_drug_x = F.relu(protein_drug_x)
        protein_drug_x = self.FClayer2(protein_drug_x)
        protein_drug_x = F.relu(protein_drug_x)
        protein_drug_x = self.FClayer3(protein_drug_x)
        protein_drug_x = protein_drug_x.squeeze(-1)
        return protein_drug_x

