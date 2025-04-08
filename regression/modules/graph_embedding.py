import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.utils import get_laplacian
from torch_sparse import spmm


#  PATHWAY GRAPH EMBEDDING (Individual GNN)
class IndividualPathwayGraphEmbedding(nn.Module):
    def __init__(self, batch_size, input_dim, graph_dim, pathway_graphs):
        super(IndividualPathwayGraphEmbedding, self).__init__()
        self.batch_size = batch_size
        self.conv1 = SAGEConv(in_channels=input_dim, out_channels=graph_dim)
        self.conv2 = SAGEConv(in_channels=graph_dim, out_channels=graph_dim)

        self.cached_batched_graphs = []
        for pathway_graph in pathway_graphs:
            repeated_graphs = [pathway_graph.clone() for _ in range(batch_size)]
            batched_graph = Batch.from_data_list(repeated_graphs)
            self.cached_batched_graphs.append(batched_graph)

    def forward(self, gene_emb, pathway_idx):
        """
        Args:
            gene_emb: [BATCH_SIZE, MAX_GENE, EMB_DIM]
            pathway_idx: index for which pathway graph to use
        """
        device = gene_emb.device
        current_batch_size = gene_emb.size(0)

        # 1) 그래프의 유효 노드 수 가져오기
        base_graph = self.cached_batched_graphs[pathway_idx].to_data_list()[0]  
        num_nodes_in_graph = base_graph.num_nodes  # 예: 120

        # 2) gene_emb에서 앞쪽 num_nodes_in_graph만 슬라이싱
        gene_emb = gene_emb[:, :num_nodes_in_graph, :]  # [B, num_nodes_in_graph, E]

        if torch.isnan(gene_emb).any():
            print("NaN found")
            print("gene_embed : ", gene_emb)
            raise ValueError(f"Still NaN in gene_emb after slice: shape={gene_emb.shape}")

        # 3) (배치 그래프) 가져오기
        if current_batch_size == self.batch_size:
            batched_graph = self.cached_batched_graphs[pathway_idx]
        else:
            repeated_graphs = [base_graph.clone() for _ in range(current_batch_size)]
            batched_graph = Batch.from_data_list(repeated_graphs)

        batched_graph = batched_graph.to(device)

        # 4) 노드 피처 업데이트:  [B, num_nodes, E] -> [B*num_nodes, E]
        batched_graph.x = gene_emb.reshape(-1, gene_emb.size(-1))

        # 5) GraphSage
        x = self.conv1(batched_graph.x, batched_graph.edge_index)
        x = F.gelu(x)
        x = self.conv2(x, batched_graph.edge_index)

        # 6) Global mean pooling
        graph_embeddings = global_mean_pool(x, batched_graph.batch)  # [B, graph_dim]

        return graph_embeddings
    
# PATHWAY GRAPH EMBEDDING (Unified GNN)
class PathwayGraphEmbedding(nn.Module):
    def __init__(self, batch_size, input_dim, graph_dim, base_graphs, device):
        super(PathwayGraphEmbedding, self).__init__()
        self.num_pathways = len(base_graphs)
        self.base_graphs = base_graphs  # 각 pathway의 base graph
        self.input_dim = input_dim
        self.graph_dim = graph_dim
        self.batch_size = batch_size
        self.device = device
        self.conv1 = SAGEConv(in_channels=input_dim * 2, out_channels=graph_dim).to(self.device)
        self.conv2 = SAGEConv(in_channels=graph_dim, out_channels=graph_dim).to(self.device)

    def build_prebatched_graph(self, B):
        data_list = []
        for base_graph in self.base_graphs:
            for b in range(B):
                data = Data(x=base_graph.x, edge_index=base_graph.edge_index)
                data_list.append(data)
        batched_graph = Batch.from_data_list(data_list)
        return batched_graph

    def forward(self, gene2sub_out):
         # gene2sub_out.shape : [B, num_pathways, max_gene, emb_dim]
        B, num_pathways, max_gene, emb_dim = gene2sub_out.size()

        batched_graph = self.build_prebatched_graph(B).to(self.device)

        # 노드 피처 업데이트
        offset = 0
        x = batched_graph.x.clone()
        for i, base_graph in enumerate(self.base_graphs):
            num_nodes = base_graph.num_nodes
            for b in range(B):
                gene_emb = gene2sub_out[b, i, :num_nodes, :]
                batched_graph.x[offset : offset + num_nodes] = gene_emb
                offset += num_nodes

        # 라플라시안 연산: L @ x
        edge_index, edge_weight = get_laplacian(batched_graph.edge_index, normalization='sym', num_nodes=x.size(0))
        lap_x = spmm(edge_index, edge_weight, x.size(0), x.size(0), x)  # [N, emb_dim]
        x_combined = torch.cat([x, lap_x], dim=-1)  # [N, emb_dim * 2]
        x = self.conv1(x_combined, batched_graph.edge_index)

        # GraphSAGE GNN 적용
        x = self.conv1(batched_graph.x, batched_graph.edge_index) # (주석 처리)
        x = F.gelu(x)
        x = self.conv2(x, batched_graph.edge_index)

        # Global mean pooling
        graph_emb = global_mean_pool(x, batched_graph.batch)  # [B * Num_Pathways, graph_dim]
        graph_emb = graph_emb.view(B, num_pathways, self.graph_dim) # [B, Num_Pathways, graph_dim]

        return graph_emb

#  DRUG GRAPH EMBEDDING
class DrugGraphEmbedding(nn.Module):
    def __init__(self, input_dim, graph_dim):
        super(DrugGraphEmbedding, self).__init__()
        self.conv1 = SAGEConv(in_channels=input_dim * 2, out_channels=graph_dim) 
        self.conv2 = SAGEConv(in_channels=graph_dim, out_channels=graph_dim)

    def forward(self, drug_graph, sub2gene_out):
        """
        Args:
            drug_graph (Batch): Batched PyG Data object (each sample is a drug graph).
            sub2gene_out (Tensor): [BATCH_SIZE, MAX_SUBSTRUCTURES, EMBEDDING_DIM]
            mapped_num_subs (list[int]): 각 배치마다 "유효 substructure 노드" 수.
        """
        device = sub2gene_out.device
        B = sub2gene_out.size(0)

        # 노드 피처 업데이트
        all_node_features = []
        for b in range(B):
            num_nodes_b = drug_graph[b].num_nodes
            emb_b = sub2gene_out[b, :num_nodes_b, :]  # [num_nodes_b, E]
            all_node_features.append(emb_b)

        cat_node_features = torch.cat(all_node_features, dim=0)
        drug_graph.x = cat_node_features.to(device)

        # 라플라시안 연산: L @ x
        edge_index, edge_weight = get_laplacian(drug_graph.edge_index, normalization='sym', num_nodes=drug_graph.x.size(0))
        lap_x = spmm(edge_index, edge_weight, drug_graph.x.size(0), drug_graph.x.size(0), drug_graph.x)
        x_combined = torch.cat([drug_graph.x, lap_x], dim=-1)
        x = self.conv1(x_combined, drug_graph.edge_index)

        # Graph Sage
        x = self.conv1(drug_graph.x, drug_graph.edge_index) # (주석 처리)
        x = F.gelu(x)
        x = self.conv2(x, drug_graph.edge_index)

        # global mean pooling
        graph_embedding = global_mean_pool(x, drug_graph.batch)  # [B, graph_dim]

        return graph_embedding
