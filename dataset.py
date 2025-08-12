import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

#  [1] DATASET
class DrugResponseDataset(Dataset):
    def __init__(self, gene_embeddings, drug_embeddings, drug_substructure_embeddings, drug_substructure_masks, labels, sample_indices, **kwargs):
        
        self.gene_embeddings = gene_embeddings
        self.drug_embeddings = drug_embeddings
        self.drug_substructure_embeddings = drug_substructure_embeddings # [L, 768]
        self.drug_substructure_masks = drug_substructure_masks # [L]
        self.labels = labels
        self.sample_indices = sample_indices

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        cell_line_id, drug_id = self.sample_indices[idx]
        gene_embedding = self.gene_embeddings[cell_line_id]  # shape: [1692, 10]
        drug_embedding = self.drug_embeddings[drug_id]  # shape: [1, 768]
        drug_substructure_embedding = self.drug_substructure_embeddings[drug_id]  # shape: [L, 768]
        drug_substructure_mask = self.drug_substructure_masks[drug_id]  # shape: [L]
        label = self.labels[cell_line_id, drug_id]

        return {
            'gene_embedding': gene_embedding,
            'drug_embedding': drug_embedding,
            'drug_substructure_embedding': drug_substructure_embedding,
            'drug_substructure_mask': drug_substructure_mask,
            'label': label,
            'sample_index': (cell_line_id, drug_id) 
        }

#  [2] COLLATE FUNCTION
def collate_fn(batch):
    gene_embeddings = []
    drug_embeddings = []
    drug_substructure_embeddings = []
    drug_substructure_masks = []
    labels = []
    sample_indices = []
    
    for item in batch:
        gene_embeddings.append(item['gene_embedding'])
        drug_embeddings.append(item['drug_embedding'])
        drug_substructure_embeddings.append(item['drug_substructure_embedding'])
        drug_substructure_masks.append(item['drug_substructure_mask'])
        labels.append(item['label'])
        sample_indices.append(item['sample_index'])

    return {
        'gene_embeddings': torch.stack(gene_embeddings),
        'drug_embeddings': torch.stack(drug_embeddings),  # [B, 1, 768]
        'drug_substructure_embeddings': torch.stack(drug_substructure_embeddings),  # [B, L, 768]
        'drug_substructure_masks': torch.stack(drug_substructure_masks),  # [B, L]
        'labels': torch.tensor(labels, dtype=torch.float32),
        'sample_indices': sample_indices
    }