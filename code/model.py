import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from itertools import product
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool

GENE_EMBEDDING_DIM = 128
SUBSTRUCTURE_EMBEDDING_DIM = 128
HIDDEN_DIM = 128
FINAL_DIM = 64
OUTPUT_DIM = 1

NUM_PATHWAYS = 314
NUM_GENES = 3848
NUM_DRUGS = 78
NUM_SUBSTRUCTURES = 194

#  [1] EMBEDDING LAYERS
class GeneEmbeddingLayer(nn.Module):
    def __init__(self, num_pathways=NUM_PATHWAYS, num_genes=NUM_GENES, embedding_dim=GENE_EMBEDDING_DIM):
        super(GeneEmbeddingLayer, self).__init__()
        self.linear = nn.Linear(1, embedding_dim)  
        self.num_genes = num_genes
        self.num_pathways = num_pathways
        
    def forward(self, gene_values):
        gene_values = gene_values.view(-1, 1) # [BATCH_SIZE * NUM_PATHWAYS * NUM_GENES, 1]
        embedded_values = self.linear(gene_values)  # [BATCH_SIZE * NUM_PATHWAYS * NUM_GENES, GENE_EMBEDDING_DIM]
        return embedded_values.view(-1, self.num_pathways, self.num_genes, GENE_EMBEDDING_DIM) 

class SubstructureEmbeddingLayer(nn.Module):
    def __init__(self, num_substructures=NUM_SUBSTRUCTURES, embedding_dim=SUBSTRUCTURE_EMBEDDING_DIM):
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
    def __init__(self, input_dim, hidden_dim):
        super(PathwayGraphEmbedding, self).__init__()
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
        batch_size = gene_emb.size(0)  # Extract batch size
        global_ids = pathway_graph.global_ids  # Node IDs to update

        # Repeat the graph `batch_size` times (데이터 자체에서 graph를 복사해서 넣어주면 삭제 가능하나 메모리 상황 보고 결정)
        repeated_graphs = [pathway_graph.clone() for _ in range(batch_size)]
        batched_graph = Batch.from_data_list(repeated_graphs)

        # Update node features for all graphs
        node_features = gene_emb[:, global_ids, :]  # Shape: [BATCH_SIZE, NUM_NODES, 128]
        node_features = node_features.reshape(-1, node_features.size(-1))  # Flatten for all graphs
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
        all_node_features = []
        updated_node_features = torch.mean(drug_graph_embedding, dim=1)

        for batch_idx in range(updated_node_features.size(0)):
            global_ids = drug_graph[batch_idx].global_ids
            node_indices = torch.where(drug_graph.batch == batch_idx)[0]
            assert len(global_ids) == len(node_indices), "Mismatch between global IDs and node indices length"

            node_features = torch.zeros((len(node_indices), updated_node_features.size(-1)), device=updated_node_features.device)
            for local_idx, global_id in enumerate(global_ids):
                if global_id < updated_node_features.size(1):
                    node_features[local_idx] = updated_node_features[batch_idx, global_id]

            all_node_features.append(node_features)

        new_node_features = torch.cat(all_node_features, dim=0)
        drug_graph.x = new_node_features

        # Handle empty edge_index
        if drug_graph.edge_index is None or drug_graph.edge_index.numel() == 0:
            num_nodes = drug_graph.num_nodes
            if num_nodes > 0:
                drug_graph.edge_index = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0)
                print(f"Warning: Edge index was empty. Added self-loops for {num_nodes} nodes.")
            else:
                raise ValueError("Graph has no nodes. Cannot process empty graph.")

        # Single-node graph handling
        if drug_graph.num_nodes == 1:
            return torch.mean(drug_graph.x, dim=0, keepdim=True)

        x = self.conv1(drug_graph.x, drug_graph.edge_index)
        x = F.relu(x)
        x = self.conv2(x, drug_graph.edge_index)
        graph_embedding = global_mean_pool(x, drug_graph.batch)

        return graph_embedding


#  [4] DRUG RESPONSE MODEL
class DrugResponseModel(nn.Module):
    def __init__(self, num_pathways=NUM_PATHWAYS, num_genes=NUM_GENES, num_substructures=NUM_SUBSTRUCTURES, hidden_dim=HIDDEN_DIM, final_dim=FINAL_DIM):
        super(DrugResponseModel, self).__init__()
        self.gene_embedding_layer = GeneEmbeddingLayer(num_pathways, num_genes, GENE_EMBEDDING_DIM)
        self.substructure_embedding_layer = SubstructureEmbeddingLayer(num_substructures, SUBSTRUCTURE_EMBEDDING_DIM)
        
        self.Gene2Sub_cross_attention = CrossAttention(query_dim=GENE_EMBEDDING_DIM, key_dim=SUBSTRUCTURE_EMBEDDING_DIM)
        self.Sub2Gene_cross_attention = CrossAttention(query_dim=SUBSTRUCTURE_EMBEDDING_DIM, key_dim=GENE_EMBEDDING_DIM)

        self.pathway_graph = PathwayGraphEmbedding(GENE_EMBEDDING_DIM, hidden_dim)
        self.drug_graph = DrugGraphEmbedding(SUBSTRUCTURE_EMBEDDING_DIM, hidden_dim)

        self.fc1 = nn.Linear(GENE_EMBEDDING_DIM + hidden_dim, final_dim)
        self.fc2 = nn.Linear(final_dim, OUTPUT_DIM)
        self.sigmoid = nn.Sigmoid()

    def forward(self, gene_embeddings, pathway_graphs, substructure_embeddings, drug_graphs):

        # Gene and Substructure Embeddings
        substructure_embeddings = substructure_embeddings.int() 
        gene_embeddings = self.gene_embedding_layer(gene_embeddings)  # [Batch, Pathway, Gene, Embedding_dim]
        substructure_embeddings = self.substructure_embedding_layer(substructure_embeddings)  

        # Cross Attention (Vectorized)
        gene_query = self.Gene2Sub_cross_attention(gene_embeddings, substructure_embeddings)  # [Batch, Pathway, Gene, Embedding_dim]
        sub_query = self.Sub2Gene_cross_attention(substructure_embeddings, gene_embeddings)  # [Batch, Pathway, Substructure, Embedding_dim]

        # Graph Embedding Loop for Pathways
        pathway_graph_embeddings = torch.stack([
            self.pathway_graph(gene_query[:, i, :, :], pathway_graphs[i])
            for i in range(gene_query.size(1))
        ], dim=1)
        final_pathway_embedding = torch.mean(pathway_graph_embeddings, dim=1)  # [Batch, Embedding_dim]

        # Graph Embedding for Drugs
        final_drug_embedding = self.drug_graph(drug_graphs, sub_query)  # [Batch, hidden_dim]

        # Concatenate and Predict
        combined_embedding = torch.cat((final_pathway_embedding, final_drug_embedding), dim=-1)  # [B, Dg + H]

        x = self.fc1(combined_embedding)
        x = self.fc2(x)

        return self.sigmoid(x)