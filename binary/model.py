import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from modules.embedding_layer import GeneEmbeddingLayer, OneHotSubstructureEmbeddingLayer, ChemBERTSubstructureEmbeddingLayer
from modules.cross_attn import Gene2SubCrossAttn, Sub2GeneCrossAttn
from modules.diff_cross_attn import Gene2SubDifferCrossAttn, Sub2GeneDifferCrossAttn
from modules.graph_embedding import PathwayGraphEmbedding, UnifiedPathwayGraphEmbedding, DrugGraphEmbedding

#  DRUG RESPONSE MODEL
class DrugResponseModel(nn.Module):
    def __init__(self, pathway_graphs, pathway_masks, num_pathways, num_genes, num_substructures, gene_dim, substructure_dim, embedding_dim, hidden_dim, final_dim, output_dim, batch_size, is_differ, depth, save_intervals, file_name):
        super(DrugResponseModel, self).__init__()
        self.num_pathways = num_pathways
        self.save_intervals = save_intervals
        self.file_name = file_name
        
        self.pathway_masks = pathway_masks # [Pathway_num, Max_Gene]

        self.gene_embedding_layer = GeneEmbeddingLayer(num_pathways, num_genes, gene_dim)
        self.substructure_embedding_layer = ChemBERTSubstructureEmbeddingLayer(num_substructures, embedding_dim)
        
        if(is_differ):
            self.Gene2Sub_cross_attention = Gene2SubDifferCrossAttn(gene_embed_dim = gene_dim, sub_embed_dim = substructure_dim, depth = depth)
            self.Sub2Gene_cross_attention = Sub2GeneDifferCrossAttn(sub_embed_dim = substructure_dim, gene_embed_dim = gene_dim, depth = depth)

        else:
            self.Gene2Sub_cross_attention = Gene2SubCrossAttn(gene_dim, substructure_dim)
            self.Sub2Gene_cross_attention = Sub2GeneCrossAttn(substructure_dim, gene_dim)
        
        self.pathway_graph = UnifiedPathwayGraphEmbedding(batch_size, embedding_dim, hidden_dim, pathway_graphs)
        self.drug_graph = DrugGraphEmbedding(embedding_dim, hidden_dim)

        self.fc1 = nn.Linear(embedding_dim + hidden_dim, final_dim)
        # self.bn_fc1 = nn.BatchNorm1d(final_dim)
        self.fc2 = nn.Linear(final_dim, output_dim)

    def forward(self, gene_embeddings, drug_embeddings, drug_graphs, drug_masks):
        batch_size = gene_embeddings.size(0)

        pathway_masks = self.pathway_masks.unsqueeze(0).expand(batch_size, -1, -1) # [Batch, Pathway_num, Max_Gene]
        drug_masks = drug_masks  # [Batch, Max_Sub]

        # Gene Embeddings Layer
        gene_embeddings = self.gene_embedding_layer(gene_embeddings)  # [Batch, Pathway_num, Max_Gene, Embedding_dim]

        # Gene2Sub Cross Attention
        # Out) [Batch, Pathway_num, Max_Gene, Embedding_dim], Weight) [Batch, Pathway_num, Max_Gene, Max_Sub]
        gene2sub_out, gene2sub_weights = self.Gene2Sub_cross_attention(
            query = gene_embeddings,         # [Batch, Pathway_num, Max_Gene, Embedding_dim]
            key = drug_embeddings,   # [Batch, Max_Sub, Embedding_dim]
            query_mask = pathway_masks,      # [Batch, Pathway_num, Max_Gene]
            key_mask = drug_masks            # [Batch, Max_Sub]
        )

        # Sub2Gene Cross Attention
        # Out) [Batch, Max_Sub, Embedding_dim], Weight) [Batch, Max_Sub, Pathway_num, Max_Gene]
        sub2gene_out, sub2gene_weights = self.Sub2Gene_cross_attention(
            query = drug_embeddings,     # [Batch, Max_Sub, Embedding_dim]
            key = gene_embeddings,               # [Batch, Pathway_num, Max_Gene, Embedding_dim]
            query_mask = drug_masks,             # [Batch, Max_Sub]
            key_mask = pathway_masks             # [Batch, Pathway_num, Max_Gene]
        )

        # Substructure Embeddings Layer
        sub2gene_out = self.substructure_embedding_layer(drug_embeddings) 

        # Pathway Graph Embedding
        pathway_graph_embedding = self.pathway_graph(gene2sub_out) # [Batch, Num_Pathways, Embedding_dim]

        # Drug Graph Embedding
        drug_graph_embedding = self.drug_graph(drug_graphs, sub2gene_out) # [Batch, Embedding_dim]

        # Final Embedding
        final_pathway_embedding = torch.mean(pathway_graph_embedding, dim=1)  # [Batch, Embedding_dim]
        final_drug_embedding = drug_graph_embedding
        
        # Concatenate and Predict
        combined_embedding = torch.cat((final_pathway_embedding, final_drug_embedding), dim=-1)  # [B, Dg + H]

        x = self.fc1(combined_embedding)
        x = self.bn_fc1(x)  # BatchNorm 적용
        x = F.relu(x)
        x = self.fc2(x)

        return x, gene2sub_weights, sub2gene_weights, final_pathway_embedding, final_drug_embedding