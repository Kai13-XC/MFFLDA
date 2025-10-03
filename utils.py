

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
def multiomics_data():
    fea1 = np.genfromtxt("./lncRNA_fea/Feature_i.csv", delimiter=' ', dtype=np.dtype(str))
    fea3 = np.genfromtxt("./lncRNA_fea/Protpar.csv", delimiter=' ', dtype=np.dtype(str))
    fea4 = np.genfromtxt("./lncRNA_fea/CTD_sequence.csv", delimiter=' ', dtype=np.dtype(str))
    fea5 = np.genfromtxt("./lncRNA_fea/GC_related.csv", delimiter=' ', dtype=np.dtype(str))
    rna_i = np.hstack((fea1, fea4, fea5, fea3))

    protein = rna_i[:,0]
    rna_i_number = len(protein)
    rna_i = np.array(rna_i)
    rna_i = scale(np.array(rna_i[:, 0:], dtype=float))
    # 保留小数点后8位
    rna_i = np.round(rna_i, decimals=4)
    pca = PCA(n_components=64)
    rna_i = pca.fit_transform(rna_i)
    rnai_adj = sim_graph(rna_i, rna_i_number).astype(int)


    PC = np.genfromtxt("./drug_fea/fea_PubChem.csv", delimiter=',', dtype=np.dtype(str))
    drug = PC[:, 0]
    drug_number = len(drug)
    PC = np.array(PC)
    PC = scale(np.array(PC[:, 0:], dtype=float))
    pca = PCA(n_components=16)
    PC = pca.fit_transform(PC)
    PC_adj = sim_graph(PC, drug_number)

    ECFP = np.genfromtxt("./drug_fea/fea_ECFP.csv", delimiter=',', dtype=np.dtype(str))
    drug = ECFP[:, 0]
    drug_number = len(drug)
    ECFP = np.array(ECFP)
    ECFP = scale(np.array(ECFP[:, 0:], dtype=float))
    pca = PCA(n_components=16)
    ECFP = pca.fit_transform(ECFP)
    ECFP_adj = sim_graph(ECFP, drug_number)

    FCFP = np.genfromtxt("./drug_fea/fea_FCFP.csv", delimiter=',', dtype=np.dtype(str))
    drug = FCFP[:, 0]
    drug_number = len(drug)
    FCFP = np.array(FCFP)
    FCFP = scale(np.array(FCFP[:, 0:], dtype=float))
    pca = PCA(n_components=16)
    FCFP = pca.fit_transform(FCFP)
    FCFP_adj = sim_graph(FCFP, drug_number)
    #
    MACCS = np.genfromtxt("./drug_fea/fea_MACCS.csv", delimiter=',', dtype=np.dtype(str))
    drug = MACCS[:, 0]
    drug_number = len(drug)
    MACCS = np.array(MACCS)
    MACCS = scale(np.array(MACCS[:, 0:], dtype=float))
    pca = PCA(n_components=16)
    MACCS = pca.fit_transform(MACCS)
    MACCS_adj = sim_graph(MACCS, drug_number)

    fusion_drug_fea = np.concatenate((PC, MACCS, ECFP, FCFP), axis=1)

    fusion_drug_adj = np.logical_or(PC_adj, MACCS_adj)
    fusion_drug_adj = np.logical_or(fusion_drug_adj, ECFP_adj)
    fusion_drug_adj = np.logical_or(fusion_drug_adj, FCFP_adj)

    fusion_drug_adj = fusion_drug_adj.astype(int)

    df = pd.read_csv('shuffled_labels.csv', header=None)
    labellist = df.values.tolist()
    labellist = torch.Tensor(labellist)
    lncRNA_feat, lncRNA_adj = torch.FloatTensor(rna_i), torch.FloatTensor(rnai_adj)
    drug_feat, drug_adj = torch.FloatTensor(fusion_drug_fea), torch.FloatTensor(fusion_drug_adj)
    return lncRNA_feat, lncRNA_adj, drug_feat, drug_adj, torch.tensor(df.values.astype(np.float32))


def sim_graph(omics_data, protein_number):
    sim_matrix = np.zeros((protein_number, protein_number), dtype=float)
    adj_matrix = np.zeros((protein_number, protein_number), dtype=float)

    for i in range(protein_number):
        for j in range(i + 1):
            sim_matrix[i, j] = np.dot(omics_data[i], omics_data[j]) / (
                        np.linalg.norm(omics_data[i]) * np.linalg.norm(omics_data[j]))
            sim_matrix[j, i] = sim_matrix[i, j]

    for i in range(protein_number):
        topindex = np.argsort(sim_matrix[i])[-10:]
        for j in topindex:
            adj_matrix[i, j] = 1
    return adj_matrix

