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
        
    def forward(self, gene_values, mask=None):

        gene_values = gene_values.view(-1, 1) # [BATCH_SIZE * NUM_PATHWAYS * NUM_GENES, 1]
        embedded_values = self.linear(gene_values)  # [BATCH_SIZE * NUM_PATHWAYS * NUM_GENES, GENE_EMBEDDING_DIM]
        embedded_values = embedded_values.view(-1, self.num_pathways, self.num_genes, self.embedding_dim) 
        # [Batch, Pathway_num, Max_Gene]
        # if mask is not None:
            # mask = mask.to(embedded_values.device) 
            # embedded_values = embedded_values * mask.unsqueeze(-1)

        return embedded_values

#  SUBSTRUCTURE EMBEDDING LAYERS
class OneHotSubstructureEmbeddingLayer(nn.Module):
    # In)  [BATCH_SIZE, NUM_SUBSTRUCTURES]
    # Out) [BATCH_SIZE, NUM_SUBSTRUCTURES, SUBSTRUCTURES_EMBEDDING_DIM]
    
    def __init__(self, num_substructures, embedding_dim):
        super(OneHotSubstructureEmbeddingLayer, self).__init__()
        # self.embedding = nn.Embedding(num_substructures, embedding_dim) # [NUM_SUBSTRUCTURES, SUBSTRUCTURES_EMBEDDING_DIM]
        self.linear = nn.Linear(1, embedding_dim)  
        self.num_substructures = num_substructures
        self.embedding_dim = embedding_dim


    def forward(self, substructure_indices):        
        # embeddings = self.embedding(substructure_indices)
        substructure_indices = substructure_indices.float()
        substructure_indices = substructure_indices.view(-1, 1)
        embeddings = self.linear(substructure_indices)
        embeddings = embeddings.view(-1, self.num_substructures, self.embedding_dim) 

        return embeddings 
    

# After Cross Attn
class ChemBERTSubstructureEmbeddingLayer(nn.Module):
    # In)  [BATCH_SIZE, NUM_SUBSTRUCTURES, 768]
    # Out) [BATCH_SIZE, NUM_SUBSTRUCTURES, SUBSTRUCTURES_EMBEDDING_DIM]
    
    def __init__(self, num_substructures, embedding_dim):
        super(ChemBERTSubstructureEmbeddingLayer, self).__init__()
        self.linear = nn.Linear(768, embedding_dim)  # ChemBERT dim = 768
        self.num_substructures = num_substructures
        self.embedding_dim = embedding_dim


    def forward(self, substructure_features):        
        batch_size = substructure_features.shape[0]
        substructure_features = substructure_features.view(-1, 768) # [BATCH_SIZE*NUM_SUBSTRUCTURES, 768]
        embeddings = self.linear(substructure_features) # [BATCH_SIZE*NUM_SUBSTRUCTURES, SUBSTRUCTURES_EMBEDDING_DIM]
        embeddings = embeddings.view(batch_size, self.num_substructures, self.embedding_dim)  # [BATCH_SIZE, NUM_SUBSTRUCTURES, SUBSTRUCTURES_EMBEDDING_DIM]
        return embeddings 