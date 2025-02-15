import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

#  DATASET
class DrugResponseDataset(Dataset):
    def __init__(self, gene_embeddings, drug_embeddings, drug_graphs, drug_masks, labels, sample_indices):
        self.gene_embeddings = gene_embeddings  # {cell_line: [313, 264]}

        self.drug_embeddings = drug_embeddings  #  {drug: [17]}
        self.drug_graphs = drug_graphs  # Drug graphs
        self.drug_masks = drug_masks  # {drug: [17]}

        self.labels = labels  # {cell_line_id, drug_id : [1]}
        self.sample_indices = sample_indices  # [(cell_line_id, drug_id)]

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        cell_line_id, drug_id = self.sample_indices[idx]

        # Gene embeddings for the cell line
        gene_embedding = self.gene_embeddings[cell_line_id]  # [Pathway_num, Max_Gene]

        # Substructure embeddings for pathways
        drug_embedding = self.drug_embeddings[drug_id]  # [1, Max_Sub]
        drug_graph = self.drug_graphs[drug_id]  # Drug graph
        drug_masks = self.drug_masks[drug_id]  # [1, Max_Sub]

        # Response label for the cell line-drug pair
        label = self.labels[cell_line_id, drug_id]  # [1]

        return {
            'gene_embedding': gene_embedding,  # [313, 264]
            'drug_embedding': drug_embedding,  # [1, 17]
            'drug_graph': drug_graph,  # PyTorch Geometric Data object
            'drug_masks' : drug_masks, # [1, 17]
            'label': label,  # [1]
            'sample_index': (cell_line_id, drug_id) 
        }

#  [2] COLLATE FUNCTION
def collate_fn(batch):
    gene_embeddings = []
    drug_embeddings = []
    drug_graphs = []
    drug_masks = []
    labels = []
    sample_indices = []

    
    for item in batch:
        gene_embeddings.append(item['gene_embedding'])
        drug_embeddings.append(item['drug_embedding'])
        drug_graphs.append(item['drug_graph'])
        drug_masks.append(item['drug_masks'])
        labels.append(item['label'])
        sample_indices.append(item['sample_index'])

    drug_batch = Batch.from_data_list(drug_graphs)

    return {
        'gene_embeddings': torch.stack(gene_embeddings),  # [Batch, Pathway_num, Max_Gene]
        'drug_embeddings': torch.stack(drug_embeddings),  # [Batch, Max_Sub]
        'drug_masks': torch.stack(drug_masks, dim=0),
        'drug_graphs': drug_batch, # PyTorch Geometric Batch
        'labels': torch.tensor(labels, dtype=torch.float32),  # [Batch]
        'sample_indices': sample_indices # [Batch]
    }