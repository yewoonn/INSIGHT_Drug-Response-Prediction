import torch.nn as nn

#  GENE EMBEDDING LAYERS
class GeneEmbeddingLayer(nn.Module):
    # In)  [BATCH_SIZE, NUM_PATHWAYS, NUM_GENES] 
    # Out) [BATCH_SIZE, NUM_PATHWAYS, NUM_GENES, GENE_EMBEDDING_DIM]

    def __init__(self, embedding_dim):
        super(GeneEmbeddingLayer, self).__init__()
        self.linear = nn.Linear(1, embedding_dim)  
        
    def forward(self, gene_values):
        gene_values = gene_values.unsqueeze(-1)
        embedded_values = self.linear(gene_values)  # [BATCH_SIZE, NUM_PATHWAYS, NUM_GENES, embedding_dim]
        return embedded_values
    
#  SUBSTRUCTURE EMBEDDING LAYERS with ChemBERT
class SubstructureEmbeddingLayer(nn.Module):
    # In)  [BATCH_SIZE, NUM_SUBSTRUCTURES, 768]
    # Out) [BATCH_SIZE, NUM_SUBSTRUCTURES, SUBSTRUCTURES_EMBEDDING_DIM]
    
    def __init__(self, embedding_dim):
        super(SubstructureEmbeddingLayer, self).__init__()
        self.linear = nn.Linear(768, embedding_dim)  # ChemBERT dim = 768


    def forward(self, substructure_features):    
        embeddings = self.linear(substructure_features) # [BATCH_SIZE, NUM_SUBSTRUCTURES, SUBSTRUCTURES_EMBEDDING_DIM]

        return embeddings 