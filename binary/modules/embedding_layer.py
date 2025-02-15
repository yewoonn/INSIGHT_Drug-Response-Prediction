import torch.nn as nn

#  GENE EMBEDDING LAYERS
class GeneEmbeddingLayer(nn.Module):
    # In)  [BATCH_SIZE, NUM_PATHWAYS, NUM_GENES] 
    # Out) [BATCH_SIZE, NUM_PATHWAYS, NUM_GENES, GENE_EMBEDDING_DIM]

    def __init__(self, num_pathways, num_genes, embedding_dim):
        super(GeneEmbeddingLayer, self).__init__()
        self.linear = nn.Linear(1, embedding_dim)  
        self.num_pathways = num_pathways
        self.num_genes = num_genes
        self.embedding_dim = embedding_dim
        
    def forward(self, gene_values):

        gene_values = gene_values.view(-1, 1) # [BATCH_SIZE * NUM_PATHWAYS * NUM_GENES, 1]
        embedded_values = self.linear(gene_values)  # [BATCH_SIZE * NUM_PATHWAYS * NUM_GENES, GENE_EMBEDDING_DIM]

        return embedded_values.view(-1, self.num_pathways, self.num_genes, self.embedding_dim) 

#  SUBSTRUCTURE EMBEDDING LAYERS
class SubstructureEmbeddingLayer(nn.Module):
    # In)  [BATCH_SIZE, NUM_SUBSTRUCTURES]
    # Out) [BATCH_SIZE, NUM_SUBSTRUCTURES, SUBSTRUCTURES_EMBEDDING_DIM]
    
    def __init__(self, num_substructures, embedding_dim):
        super(SubstructureEmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(num_substructures, embedding_dim) # [NUM_SUBSTRUCTURES, SUBSTRUCTURES_EMBEDDING_DIM]

    def forward(self, substructure_indices):
        return self.embedding(substructure_indices) 