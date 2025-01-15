import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from itertools import product
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
import math
import os

from DiffTransformer.rms_norm import RMSNorm


#  [1] EMBEDDING LAYERS
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

class SubstructureEmbeddingLayer(nn.Module):
    # In)  [BATCH_SIZE, NUM_SUBSTRUCTURES]
    # Out) [BATCH_SIZE, NUM_SUBSTRUCTURES, SUBSTRUCTURES_EMBEDDING_DIM]
    
    def __init__(self, num_substructures, embedding_dim):
        super(SubstructureEmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(num_substructures, embedding_dim) # [NUM_SUBSTRUCTURES, SUBSTRUCTURES_EMBEDDING_DIM]

    def forward(self, substructure_indices):
        return self.embedding(substructure_indices) 

#  [2] CROSS ATTENTION
class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim):
        super(CrossAttention, self).__init__()
        self.query_layer = nn.Linear(query_dim, query_dim)  
        self.key_layer = nn.Linear(key_dim, query_dim)      
        self.value_layer = nn.Linear(key_dim, query_dim)    

    def forward(self, query_embeddings, key_embeddings):
        query = self.query_layer(query_embeddings) 
        key = self.key_layer(key_embeddings)        
        value = self.value_layer(key_embeddings)   

        # Attention Scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2))  
        attention_weights = F.softmax(attention_scores, dim=-1)        

        # Apply Attention
        attended_embeddings = torch.matmul(attention_weights, value)  
        
        return attended_embeddings, attention_weights
    

class DifferCrossAttn(nn.Module):
    def __init__(self, embed_dim: int, depth: int = 0):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // 2
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)  # (E -> E)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)  # (E -> E)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)  # (E -> E)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)

        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, gene: torch.Tensor, substructure: torch.Tensor) -> torch.Tensor:
        bsz, n_gene, _ = gene.size()
        _, n_sub,  _ = substructure.size()

        # 1) Q, K, V 구하기 (Cross Attention)
        q = self.q_proj(gene)              # (B, n_gene, E)
        k = self.k_proj(substructure)      # (B, n_sub,  E)
        v = self.v_proj(substructure)      # (B, n_sub,  E)

        # 2) Q, K -> [2, head_dim], V -> [2 * head_dim]
        q = q.view(bsz, n_gene, 2, self.head_dim)   # (B, n_gene, 2, head_dim)
        k = k.view(bsz, n_sub,  2, self.head_dim)   # (B, n_sub,  2, head_dim)
        v = v.view(bsz, n_sub,  2 * self.head_dim)  # (B, n_sub,  2 * head_dim)

        # 3) matmul 위해 transpose (slot 차원을 두 번째로)
        q = q.transpose(1, 2)  # (B, 2, n_gene, head_dim)
        k = k.transpose(1, 2)  # (B, 2, n_sub,  head_dim)
        v = v.unsqueeze(1)     # (B, 1, n_sub, 2*head_dim)

        q = q * self.scaling

        # 4) attention score = Q @ K^T
        #    => (B, 2, n_gene, head_dim) x (B, 2, head_dim, n_sub) = (B, 2, n_gene, n_sub)
        attn_weights = torch.matmul(q, k.transpose(-1, -2))  # (B, 2, n_gene, n_sub)
        attn_weights = F.softmax(attn_weights, dim=-1)  # (B, 2, n_gene, n_sub)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1))
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # 5) 두 슬롯(0, 1) 간 차이를 내어 최종 attn_weights 계산
        diff_attn_weights = attn_weights[:, 0] - lambda_full * attn_weights[:, 1]  # (B, n_gene, n_sub)
        diff_attn = diff_attn_weights.unsqueeze(1)  # (B, 1, n_gene, n_sub)

        attn_output = torch.matmul(diff_attn, v)
        attn_output = self.subln(attn_output)               # (B, 1, n_gene, 2*head_dim)
        attn_output = attn_output * (1.0 - self.lambda_init)
        attn_output = attn_output.squeeze(1)                # (B, n_gene, 2*head_dim)
        out = self.out_proj(attn_output)                    # (B, n_gene, E)

        return out, diff_attn_weights
    

#  [3] GRAPH EMBEDDING
class PathwayGraphEmbedding(nn.Module):
    def __init__(self, batch_size, input_dim, hidden_dim, pathway_graphs):
        super(PathwayGraphEmbedding, self).__init__()
        self.batch_size = batch_size
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.cached_batched_graphs = []
        for pathway_graph in pathway_graphs:
            repeated_graphs = [pathway_graph.clone() for _ in range(batch_size)]
            batched_graph = Batch.from_data_list(repeated_graphs)
            self.cached_batched_graphs.append(batched_graph)

    def forward(self, gene_emb, pathway_idx):
        """
        Args:
            gene_emb: Tensor of shape [BATCH_SIZE, NUM_GENES, 128]
            pathway_graph: PyTorch Geometric Data object (single graph)
        Returns:
            graph_embeddings: Tensor of shape [BATCH_SIZE, NUM_PATHWAYS, EMBEDDING_DIM]
        """
        # Repeat the graph `batch_size` times
        current_batch_size = gene_emb.size(0)

        if current_batch_size == self.batch_size:
            batched_graph = self.cached_batched_graphs[pathway_idx]
        else:
            # Dynamically create a batched graph for the current batch size
            pathway_graph = self.cached_batched_graphs[pathway_idx].to_data_list()[0]  # Retrieve base graph
            repeated_graphs = [pathway_graph.clone() for _ in range(current_batch_size)]
            batched_graph = Batch.from_data_list(repeated_graphs)

        # Update node features for all graphs
        device = gene_emb.device
        batched_graph = batched_graph.to(device)
        batched_graph.x = gene_emb.view(-1, gene_emb.size(-1))  # Shape: [BATCH_SIZE * NUM_VALID_NODES, 128]

        # GCN layers
        x = F.relu(self.conv1(batched_graph.x, batched_graph.edge_index))
        x = self.conv2(x, batched_graph.edge_index)

        # Global mean pooling to aggregate graph embeddings
        graph_embeddings = global_mean_pool(x, batched_graph.batch)  # Shape: [BATCH_SIZE, EMBEDDING_DIM]
        
        return graph_embeddings

class DrugGraphEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DrugGraphEmbedding, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, drug_graph, drug_graph_embedding, all_global_ids, mapped_indices):
        """
        Args:
            drug_graph (Batch): Batched PyTorch Geometric Data object.
            drug_graph_embedding (Tensor): [BATCH_SIZE, NUM_SUBSTRUCTURES, EMBEDDING_DIM]
            all_global_ids (list): List of global IDs for all nodes in the batch.
            mapped_indices (list): List of indices mapping all_global_ids to set_global_ids.
        """
        # 1. Initialize list for all node features
        all_node_features = []

        # 2. Process each batch independently
        for batch_idx in range(drug_graph_embedding.size(0)):
            # Extract `global_ids` for the current batch
            batch_global_ids = drug_graph[batch_idx].global_ids

            # Map global IDs to `mapped_indices`
            batch_local_indices = [mapped_indices[all_global_ids.index(global_id)] for global_id in batch_global_ids]
            batch_local_indices_tensor = torch.tensor(batch_local_indices, device=drug_graph_embedding.device)

            # Gather node features for the current batch
            batch_node_features = drug_graph_embedding[batch_idx, batch_local_indices_tensor]
            all_node_features.append(batch_node_features)

        # 3. Concatenate all node features
        new_node_features = torch.cat(all_node_features, dim=0)  # Shape: [TOTAL_NUM_NODES, EMBEDDING_DIM]
        # print("new_node_features :", new_node_features.shape)

        # 4. Update drug_graph node features
        drug_graph.x = new_node_features

        # 5. Apply GCN layers
        x = self.conv1(drug_graph.x, drug_graph.edge_index)
        x = F.relu(x)
        x = self.conv2(x, drug_graph.edge_index)

        # 6. Perform global mean pooling
        graph_embedding = global_mean_pool(x, drug_graph.batch)  # Shape: [BATCH_SIZE, HIDDEN_DIM]

        return graph_embedding

import time

#  [4] DRUG RESPONSE MODEL
class DrugResponseModel(nn.Module):
    def __init__(self, num_pathways, pathway_graphs, pathway_genes_dict, num_genes, num_substructures, gene_dim, substructure_dim, hidden_dim, final_dim, output_dim, batch_size, is_differ, depth, save_intervals, save_pathways, file_name, attn_logger):
        super(DrugResponseModel, self).__init__()
        self.save_intervals = save_intervals
        self.save_pathways = save_pathways
        self.file_name = file_name
        self.attn_logger = attn_logger

        self.pathway_genes_dict = pathway_genes_dict

        self.gene_embedding_layer = GeneEmbeddingLayer(num_pathways, num_genes, gene_dim)
        self.substructure_embedding_layer = SubstructureEmbeddingLayer(num_substructures, substructure_dim)
        
        if(is_differ):
            self.Gene2Sub_cross_attention = DifferCrossAttn(gene_dim, depth)
            self.Sub2Gene_cross_attention = DifferCrossAttn(substructure_dim, depth)

        else:
            self.Gene2Sub_cross_attention = CrossAttention(gene_dim, substructure_dim)
            self.Sub2Gene_cross_attention = CrossAttention(substructure_dim, gene_dim)

        self.pathway_graph = PathwayGraphEmbedding(batch_size, gene_dim, hidden_dim, pathway_graphs)
        self.drug_graph = DrugGraphEmbedding(substructure_dim, hidden_dim)

        self.fc1 = nn.Linear(gene_dim + hidden_dim, final_dim)
        self.fc2 = nn.Linear(final_dim, output_dim)

    def forward(self, gene_embeddings, substructure_embeddings, drug_graphs, epoch, sample_indices):
        # Substructure Index
        all_global_ids = list(id for sublist in drug_graphs['global_ids'] for id in sublist)
        set_global_ids = sorted(set(all_global_ids))

        id_to_set_index = {global_id: idx for idx, global_id in enumerate(set_global_ids)}
        mapped_indices = [id_to_set_index[global_id] for global_id in all_global_ids]

        filtered_sub_embeddings = substructure_embeddings[:, set_global_ids]

        # Gene and Substructure Embeddings
        gene_embeddings = self.gene_embedding_layer(gene_embeddings)  # [Batch, Pathway, Gene, Embedding_dim]
        filtered_sub_embeddings = self.substructure_embedding_layer(filtered_sub_embeddings)  # [Batch, Substructure, Embedding_dim]
        
        pathway_graph_embeddings = []
        drug_embeddings = []

        # Cross Attention & Pathway Graph Embedding
        for i in range(gene_embeddings.size(1)):
            # 유효한 gene 필터링
            valid_gene_indices = self.pathway_genes_dict[i]  # [Num_Valid_Genes]
            filtered_gene_embeddings = gene_embeddings[:, i, valid_gene_indices, :]  # [Batch, Num_Valid_Genes, Embedding_dim]
            
            # Cross Attention
            gene_attention_out, gene_attention_weights = self.Gene2Sub_cross_attention(
                filtered_gene_embeddings,  # [Batch, Num_Valid_Genes, Embedding_dim]
                filtered_sub_embeddings    # [Batch, Num_Substructures, Embedding_dim]
            ) # [Batch, Num_Valid_Genes, Embedding_dim]


            sub_attention_out, sub_attention_weights = self.Sub2Gene_cross_attention(
                filtered_sub_embeddings,   # [Batch, Num_Substructures, Embedding_dim]
                filtered_gene_embeddings   # [Batch, Num_Valid_Genes, Embedding_dim]
            )

            if (epoch % self.save_intervals == 0) and (i in self.save_pathways):
                self.attn_logger.add_gene_attention(
                    sample_indices, i,
                    gene_attention_weights,
                    valid_gene_indices,
                    set_global_ids
                )
                self.attn_logger.add_sub_attention(
                    sample_indices, i,
                    sub_attention_weights,
                    valid_gene_indices,
                    set_global_ids
                )

            # Pathway Graph Embedding
            graph_embedding = self.pathway_graph(
                gene_attention_out,  
                i       
            )

            pathway_graph_embeddings.append(graph_embedding)
            drug_embeddings.append(sub_attention_out)
        
        pathway_graph_embedding = torch.stack(pathway_graph_embeddings, dim=1) # [Batch, Num_Pathways, Embedding_dim]]
        drug_embeddings = torch.mean(torch.stack(drug_embeddings, dim=1), dim=1)

        # Drug Graph Embedding
        drug_graph_embedding = self.drug_graph(
            drug_graphs,     
            drug_embeddings,
            all_global_ids,
            mapped_indices
        ) # [Batch, hidden_dim]

        # Final Embedding
        final_pathway_embedding = torch.mean(pathway_graph_embedding, dim=1)  # [Batch, Embedding_dim]
        final_drug_embedding = drug_graph_embedding
        
        # Concatenate and Predict
        combined_embedding = torch.cat((final_pathway_embedding, final_drug_embedding), dim=-1)  # [B, Dg + H]

        x = self.fc1(combined_embedding)
        x = self.fc2(x)

        return x
