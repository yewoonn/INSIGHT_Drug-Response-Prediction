import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from itertools import product
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool


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
        
        return attended_embeddings
    

#  [3] GRAPH EMBEDDING
class PathwayGraphEmbedding(nn.Module):
    def __init__(self, batch_size, input_dim, hidden_dim):
        super(PathwayGraphEmbedding, self).__init__()
        self.batch_size = batch_size
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, gene_emb, pathway_graph):
        """
        Args:
            gene_emb: Tensor of shape [BATCH_SIZE, NUM_GENES, 128]
            pathway_graph: PyTorch Geometric Data object (single graph)
        Returns:
            graph_embeddings: Tensor of shape [BATCH_SIZE, NUM_PATHWAYS, EMBEDDING_DIM]
        """
        # Repeat the graph `batch_size` times
        repeated_graphs = [pathway_graph.clone() for _ in range(self.batch_size)]
        batched_graph = Batch.from_data_list(repeated_graphs)
        expected_nodes = batched_graph.x.size(0)

        # Update node features for all graphs
        node_features = gene_emb  # Shape: [BATCH_SIZE, NUM_NODES, 128]
        node_features = gene_emb.view(-1, gene_emb.size(-1))  # Shape: [BATCH_SIZE * NUM_VALID_NODES, 128]
        assert node_features.size(0) == expected_nodes, \
            f"Node feature count mismatch: {node_features.size(0)} != {expected_nodes}"
        batched_graph.x = node_features  # Assign to graph

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

    def forward(self, drug_graph, drug_graph_embedding):
        """
        Args:
            drug_graph (Batch): Batched PyTorch Geometric Data object.
            drug_graph_embedding (Tensor): [BATCH_SIZE, NUM_PATHWAYS, NUM_SUBSTRUCTURES, EMBEDDING_DIM]
        """
        all_node_features = []
        updated_node_features = torch.mean(drug_graph_embedding, dim=1)  # [BATCH_SIZE, NUM_SUBSTRUCTURES, SUBSTRUCTURE_EMBEDDING_DIM]

        # Batch Loop
        for batch_idx in range(updated_node_features.size(0)):  
            # Get global IDs and node indices
            global_ids = drug_graph[batch_idx].global_ids  
            node_indices = torch.where(drug_graph.batch == batch_idx)[0]

            # Ensure global_ids and node_indices match in length
            assert len(global_ids) == len(node_indices), "Mismatch between global IDs and node indices length"

            # Update node features for the current batch
            node_features = torch.zeros((len(node_indices), updated_node_features.size(-1)), device=updated_node_features.device)
            for local_idx, global_id in enumerate(global_ids):
                if global_id < updated_node_features.size(1):
                    node_features[local_idx] = updated_node_features[batch_idx, global_id]

            # Append the current batch's node features to the list
            all_node_features.append(node_features)

        # updated node features from all batches
        new_node_features = torch.cat(all_node_features, dim=0)  # Shape: [TOTAL_NUM_NODES, EMBEDDING_DIM]
        drug_graph.x = new_node_features

        # GCN layers
        x = self.conv1(drug_graph.x, drug_graph.edge_index)
        x = F.relu(x)
        x = self.conv2(x, drug_graph.edge_index)

        # Perform global mean pooling
        graph_embedding = global_mean_pool(x, drug_graph.batch)  # Shape: [BATCH_SIZE, HIDDEN_DIM]
        
        return graph_embedding

#  [4] DRUG RESPONSE MODEL
class DrugResponseModel(nn.Module):
    def __init__(self, num_pathways, num_genes, num_substructures, gene_dim, substructure_dim, hidden_dim, final_dim, output_dim, batch_size):
        super(DrugResponseModel, self).__init__()
        self.gene_embedding_layer = GeneEmbeddingLayer(num_pathways, num_genes, gene_dim)
        self.substructure_embedding_layer = SubstructureEmbeddingLayer(num_substructures, substructure_dim)
        
        self.Gene2Sub_cross_attention = CrossAttention(query_dim=gene_dim, key_dim=substructure_dim)
        self.Sub2Gene_cross_attention = CrossAttention(query_dim=substructure_dim, key_dim=gene_dim)

        self.pathway_graph = PathwayGraphEmbedding(batch_size ,gene_dim, hidden_dim)
        self.drug_graph = DrugGraphEmbedding(substructure_dim, hidden_dim)

        self.fc1 = nn.Linear(gene_dim + hidden_dim, final_dim)
        self.fc2 = nn.Linear(final_dim, output_dim)

    def forward(self, gene_embeddings, pathway_graphs, substructure_embeddings, drug_graphs):
        # Gene and Substructure Embeddings
        print("gene embeddings : ", gene_embeddings.shape)
        print("substructure embeddings : ", substructure_embeddings.shape)

        substructure_embeddings = substructure_embeddings.int()  
        gene_embeddings = self.gene_embedding_layer(gene_embeddings)  # [Batch, Pathway, Gene, Embedding_dim]
        substructure_embeddings = self.substructure_embedding_layer(substructure_embeddings)  # [Batch, Substructure, Embedding_dim]

        pathway_graph_embeddings = []
        drug_embeddings = []

        # Pathway Cross Attention & Graph Embedding
        for i in range(len(pathway_graphs)):
            pathway_graph = pathway_graphs[i]

            # 유효한 gene 필터링
            valid_gene_indices = pathway_graph['global_ids']  # [Num_Valid_Genes]
            filtered_gene_embeddings = gene_embeddings[:, i, valid_gene_indices, :]  # [Batch, Num_Valid_Genes, Embedding_dim]

            # Cross Attention
            gene_attention_out = self.Gene2Sub_cross_attention(
                filtered_gene_embeddings,  # [Batch, Num_Valid_Genes, Embedding_dim]
                substructure_embeddings  # [Batch, Num_Substructures, Embedding_dim]
            )  # [Batch, Num_Valid_Genes, Embedding_dim]

            sub_attention_out = self.Sub2Gene_cross_attention(
                substructure_embeddings, # [Batch, Num_Substructures, Embedding_dim]
                filtered_gene_embeddings # [Batch, Num_Valid_Genes, Embedding_dim]
            )

            # Pathway Graph Embedding
            graph_embedding = self.pathway_graph(gene_attention_out, pathway_graph)

            pathway_graph_embeddings.append(graph_embedding)
            drug_embeddings.append(sub_attention_out)
        
        pathway_graph_embedding = torch.stack(pathway_graph_embeddings, dim=1) # [Batch, Num_Pathways, Embedding_dim]]
        drug_embeddings = torch.stack(drug_embeddings, dim = 1)

        # Drug Graph Embedding
        drug_embeddings = torch.mean(drug_embeddings, dim = 1)
        drug_graph_embedding = self.drug_graph(drug_graphs, drug_embeddings)  # [Batch, hidden_dim]

        # Final Embedding
        final_pathway_embedding = torch.mean(pathway_graph_embedding, dim=1)  # [Batch, Embedding_dim]
        final_drug_embedding = drug_graph_embedding

        # Concatenate and Predict
        combined_embedding = torch.cat((final_pathway_embedding, final_drug_embedding), dim=-1)  # [B, Dg + H]

        x = self.fc1(combined_embedding)
        x = self.fc2(x)

        return x
