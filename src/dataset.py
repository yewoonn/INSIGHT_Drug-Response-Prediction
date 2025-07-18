import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

#  DATASET
class DrugResponseDataset(Dataset):
    def __init__(self, gene_embeddings, drug_embeddings, drug_masks, labels, sample_indices, drug_spectral_embeddings): 
        self.gene_embeddings = gene_embeddings  # {cell_line: [312, 245]}

        self.drug_embeddings = drug_embeddings  #  {drug: [11]}
        self.drug_masks = drug_masks  # {drug: [11]}

        self.labels = labels  # {cell_line_id, drug_id : [1]}
        self.sample_indices = sample_indices  # [(cell_line_id, drug_id)]
        self.drug_spectral_embeddings = drug_spectral_embeddings 

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        cell_line_id, drug_id = self.sample_indices[idx]

        # Gene embeddings for the cell line
        gene_embedding = self.gene_embeddings[cell_line_id]  # [Pathway_num, Max_Gene]

        # Substructure embeddings for pathways
        drug_embedding = self.drug_embeddings[drug_id]  # [1, Max_Sub]
        drug_masks = self.drug_masks[drug_id]  # [1, Max_Sub]
        drug_spectral_embedding = self.drug_spectral_embeddings[drug_id]  

        # Response label for the cell line-drug pair
        label = self.labels[cell_line_id, drug_id]  # [1]

        return {
            'gene_embedding': gene_embedding,  # [312, 245]
            'drug_embedding': drug_embedding,  # [1, 11]
            'drug_spectral_embedding': drug_spectral_embedding, 
            'drug_masks' : drug_masks, # [1, 11]
            'label': label,  # [1]
            'sample_index': (cell_line_id, drug_id) 
        }

#  [2] COLLATE FUNCTION
def collate_fn(batch):
    gene_embeddings = []
    drug_embeddings = []
    drug_masks = []
    labels = []
    sample_indices = []
    drug_spectral_embeddings = [] 
    
    for item in batch:
        gene_embeddings.append(item['gene_embedding'])
        drug_embeddings.append(item['drug_embedding'])
        drug_spectral_embeddings.append(item['drug_spectral_embedding']) 
        drug_masks.append(item['drug_masks'])
        labels.append(item['label'])
        sample_indices.append(item['sample_index'])


    return {
        'gene_embeddings': torch.stack(gene_embeddings),  # [Batch, Pathway_num, Max_Gene]
        'drug_embeddings': torch.stack(drug_embeddings),  # [Batch, Max_Sub]
        'drug_spectral_embeddings': torch.stack(drug_spectral_embeddings),  # [Batch, Max_Sub, 4] 
        'drug_masks': torch.stack(drug_masks, dim=0),
        'labels': torch.tensor(labels, dtype=torch.float32),  # [Batch]
        'sample_indices': sample_indices # [Batch]
    }