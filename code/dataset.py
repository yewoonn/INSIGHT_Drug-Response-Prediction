import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.functional as F
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool

#  [1] DATASET
class DrugResponseDataset(Dataset):
    def __init__(self, gene_embeddings, drug_embeddings, drug_graphs, labels, sample_indices):
        """
        Args:
            gene_embeddings (dict): {cell_line_id: Tensor}, Gene embeddings for each cell line.
            drug_graphs (dict): List of PyTorch Geometric Data objects for each drug (indexed by drug_id).
            substructure_embeddings (Tensor): [245, 193], Substructure embeddings for pathways.
            labels (dict): {cell_line_id: Tensor}, Drug response labels for each cell line and drug pair.
            sample_indices (list): [(cell_line_id, drug_idx)], List of cell line and drug index pairs.
        """
        self.gene_embeddings = gene_embeddings  # {cell_line_id: [245, 231]}
        self.drug_graphs = drug_graphs  # Drug graphs
        self.drug_embeddings = drug_embeddings  # [170]
        self.labels = labels  # {cell_line_id, drug_id : [1]}
        self.sample_indices = sample_indices  # [(cell_line_id, drug_id)]


    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        cell_line_id, drug_id = self.sample_indices[idx]

        # Gene embeddings for the cell line
        gene_embedding = self.gene_embeddings[cell_line_id]  # [245, 231]

        # Substructure embeddings for pathways
        drug_embedding = self.drug_embeddings[drug_id]  # [1, 170]
        drug_graph = self.drug_graphs[drug_id]  # Drug graphs

        # Get the label for the cell line-drug pair
        label = self.labels[cell_line_id, drug_id]  # Scalar

        return {
            'gene_embedding': gene_embedding,  # [245, 231]
            'drug_embedding': drug_embedding,  # [245, 170]
            'drug_graph': drug_graph,  # PyTorch Geometric Data object
            'label': label,  # Scalar,
            'sample_index': (cell_line_id, drug_id)
        }

#  [2] COLLATE FUNCTION
def collate_fn(batch):
    gene_embeddings = []
    drug_embeddings = []
    drug_graphs = []
    labels = []
    sample_indices = [] # 추가

    
    for item in batch:
        gene_embeddings.append(item['gene_embedding'])
        drug_embeddings.append(item['drug_embedding'])
        drug_graphs.append(item['drug_graph'])
        labels.append(item['label'])
        sample_indices.append(item['sample_index']) # 추가

    drug_batch = Batch.from_data_list(drug_graphs)

    return {
        'gene_embeddings': torch.stack(gene_embeddings),  # [batch_size, num_pathways, num_genes]
        'substructure_embeddings': torch.stack(drug_embeddings),  # [batch_size, num_substructures]
        'drug_graphs': drug_batch, # PyTorch Geometric Batch
        'labels': torch.tensor(labels, dtype=torch.float32),  # [batch_size]
        'sample_indices': sample_indices # 추가
    }