import torch
import torch.nn as nn
import time

from modules.embedding_layer import GeneEmbeddingLayer, SubstructureEmbeddingLayer
from modules.cross_attn import Gene2SubCrossAttn, Sub2GeneCrossAttn
from modules.diff_cross_attn import Gene2SubDifferCrossAttn, Sub2GeneDifferCrossAttn
from modules.graph_embedding import PathwayGraphEmbedding, DrugGraphEmbedding

#  DRUG RESPONSE MODEL
class DrugResponseModel(nn.Module):
    def __init__(self, pathway_graphs, pathway_masks, num_pathways, num_genes, num_substructures, gene_dim, substructure_dim, hidden_dim, final_dim, output_dim, batch_size, is_differ, depth, save_intervals, file_name):
        super(DrugResponseModel, self).__init__()
        self.num_pathways = num_pathways
        self.save_intervals = save_intervals
        self.file_name = file_name
        
        self.pathway_masks = pathway_masks # [Pathway_num, Max_Gene]

        self.gene_embedding_layer = GeneEmbeddingLayer(num_pathways, num_genes, gene_dim)
        self.substructure_embedding_layer = SubstructureEmbeddingLayer(num_substructures, substructure_dim)
        
        if(is_differ):
            self.Gene2Sub_cross_attention = Gene2SubDifferCrossAttn(gene_dim, depth)
            self.Sub2Gene_cross_attention = Sub2GeneDifferCrossAttn(substructure_dim, depth)

        else:
            self.Gene2Sub_cross_attention = Gene2SubCrossAttn(gene_dim, substructure_dim)
            self.Sub2Gene_cross_attention = Sub2GeneCrossAttn(substructure_dim, gene_dim)
        
        self.pathway_graph = PathwayGraphEmbedding(batch_size, gene_dim, hidden_dim, pathway_graphs)
        self.drug_graph = DrugGraphEmbedding(substructure_dim, hidden_dim)

        self.fc1 = nn.Linear(gene_dim + hidden_dim, final_dim)
        self.fc2 = nn.Linear(final_dim, output_dim)

    def forward(self, gene_embeddings, drug_embeddings, drug_graphs, drug_masks, epoch, sample_indices):
        batch_size = gene_embeddings.size(0)
        # print(f"Substructure_embedding after Model Input: {drug_embeddings[0,:5]}")

        # start_time = time.time()
        # Gene and Substructure Embeddings
        gene_embeddings = self.gene_embedding_layer(gene_embeddings)  # [Batch, Pathway_num, Max_Gene, Embedding_dim]
        substructure_embeddings = self.substructure_embedding_layer(drug_embeddings)  # [Batch, Max_Sub, Embedding_dim]
        # print(f"Substructure_embedding after Substructure Embedding Layer: {substructure_embeddings[0,:5,:]}")

        # start_time = time.time()
        pathway_masks = self.pathway_masks.unsqueeze(0).expand(batch_size, -1, -1) # [Batch, Pathway_num, Max_Gene]
        drug_masks = drug_masks  # [Batch, Max_Sub]
        # print(f"Masks time: {time.time() - start_time:.4f} seconds")

        pathway_graph_embeddings = []
        drug_embeddings = []

        # Gene2Sub Cross Attention
        # Out) [Batch, Pathway_num, Max_Gene, Embedding_dim], Weight) [Batch, Pathway_num, Max_Gene, Max_Sub]
        # start_time = time.time()
        gene2sub_out, gene2sub_weights = self.Gene2Sub_cross_attention(
            query = gene_embeddings,         # [Batch, Pathway_num, Max_Gene, Embedding_dim]
            key = substructure_embeddings,   # [Batch, Max_Sub, Embedding_dim]
            mask = pathway_masks             # [Batch, Pathway_num, Max_Gene]
        )
        # print(f"Gene2Sub Cross Attention time: {time.time() - start_time:.4f} seconds")

        # Sub2Gene Cross Attention
        # Out) [Batch, Max_Sub, Embedding_dim], Weight) [Batch, Max_Sub, Pathway_num, Max_Gene]
        # start_time = time.time()
        sub2gene_out, sub2gene_weights = self.Sub2Gene_cross_attention(
            query = substructure_embeddings,    # [Batch, Max_Sub, Embedding_dim]
            key = gene_embeddings,               # [Batch, Pathway_num, Max_Gene, Embedding_dim]
            mask = drug_masks                   # [Batch, Max_Sub]
        )
        # print(f"Sub2Gene Cross Attention time: {time.time() - start_time:.4f} seconds")

        # Pathway Graph Embedding
        # start_time = time.time()
        pathway_graph_embeddings = []
        for i in range(self.num_pathways):
            gene_emb = gene2sub_out[:, i, :, :]
            graph_emb = self.pathway_graph(gene_emb, i)
            pathway_graph_embeddings.append(graph_emb)
        pathway_graph_embedding = torch.stack(pathway_graph_embeddings, dim=1) # [Batch, Num_Pathways, Embedding_dim]]
        # print(f"Pathway Graph Embedding time: {time.time() - start_time:.4f} seconds")

        # Drug Graph Embedding
        # start_time = time.time()
        drug_graph_embedding = self.drug_graph(drug_graphs, sub2gene_out) # [Batch, Embedding_dim]
        # print(f"Drug Graph Embedding time: {time.time() - start_time:.4f} seconds")

        # Final Embedding
        # start_time = time.time()
        final_pathway_embedding = torch.mean(pathway_graph_embedding, dim=1)  # [Batch, Embedding_dim]
        final_drug_embedding = drug_graph_embedding
        
        # Concatenate and Predict
        combined_embedding = torch.cat((final_pathway_embedding, final_drug_embedding), dim=-1)  # [B, Dg + H]

        x = self.fc1(combined_embedding)
        x = self.fc2(x)
        # print(f"Final Prediction time: {time.time() - start_time:.4f} seconds")

        return x, gene2sub_weights, sub2gene_weights, final_pathway_embedding, final_drug_embedding