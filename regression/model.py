import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.embedding_layer import GeneEmbeddingLayer, SubstructureEmbeddingLayer
from modules.diff_cross_attn import Gene2SubDifferCrossAttn, Sub2GeneDifferCrossAttn, Gene2SubDifferCrossMHA, Sub2GeneDifferCrossMHA

#  DRUG RESPONSE MODEL
class DrugResponseModel(nn.Module):
    def __init__(self, pathway_masks, num_pathways, gene_layer_dim, substructure_layer_dim, cross_attn_dim, final_dim, output_dim, save_intervals, file_name, device):
        super(DrugResponseModel, self).__init__()
        self.num_pathways = num_pathways
        self.save_intervals = save_intervals
        self.file_name = file_name
        self.device = device
        
        self.pathway_masks = pathway_masks # [Pathway_num, Max_Gene]

        self.gene_embedding_layer = GeneEmbeddingLayer(gene_layer_dim)
        self.substructure_embedding_layer = SubstructureEmbeddingLayer(cross_attn_dim)
        
        # self.Gene2Sub_cross_attention = Gene2SubDifferCrossAttn(gene_embed_dim = gene_layer_dim, sub_embed_dim = substructure_layer_dim) 
        self.Gene2Sub_cross_attention = Gene2SubDifferCrossMHA(gene_embed_dim = gene_layer_dim, sub_embed_dim = substructure_layer_dim, attention_dim = cross_attn_dim, num_heads = 4, depth = 1)
        # self.Sub2Gene_cross_attention = Sub2GeneDifferCrossAttn(sub_embed_dim = substructure_layer_dim, gene_embed_dim = gene_layer_dim)
        self.Sub2Gene_cross_attention = Sub2GeneDifferCrossMHA(sub_embed_dim = substructure_layer_dim, gene_embed_dim = gene_layer_dim, attention_dim = cross_attn_dim, num_heads = 4, depth = 1)
        
        self.fc1 = nn.Linear(2*cross_attn_dim, final_dim)
        self.bn_fc1 = nn.BatchNorm1d(final_dim)
        self.fc2 = nn.Linear(final_dim, output_dim)

    def forward(self, gene_embeddings, drug_embeddings, drug_masks):
        batch_size = gene_embeddings.size(0)

        pathway_masks = self.pathway_masks.unsqueeze(0).expand(batch_size, -1, -1) # [Batch, Pathway_num, Max_Gene]
        drug_masks = drug_masks  # [Batch, Max_Sub]

        # Gene and Substructure Embeddings
        gene_embeddings = self.gene_embedding_layer(gene_embeddings)  # [Batch, Pathway_num, Max_Gene, Gene_Layer_dim]
        
        # Gene2Sub Cross Attention
        # Out) [Batch, Pathway_num, Max_Gene, Cross_Attn_dim], Weight) [Batch, Pathway_num, Max_Gene, Max_Sub]
        gene2sub_out, gene2sub_weights = self.Gene2Sub_cross_attention(
            query = gene_embeddings,         # [Batch, Pathway_num, Max_Gene, Gene_Layer_dim]
            key = drug_embeddings,   # [Batch, Max_Sub, Substructure_Layer_dim]
            query_mask = pathway_masks,      # [Batch, Pathway_num, Max_Gene]
            key_mask = drug_masks            # [Batch, Max_Sub]
        )

        # Sub2Gene Cross Attention
        # Out) [Batch, Max_Sub, Cross_Attn_dim], Weight) [Batch, Max_Sub, Pathway_num, Max_Gene]
        sub2gene_out, sub2gene_weights = self.Sub2Gene_cross_attention(
            query = drug_embeddings,     # [Batch, Max_Sub, Substructure_Layer_dim]
            key = gene_embeddings,               # [Batch, Pathway_num, Max_Gene, Gene_Layer_dim]
            query_mask = drug_masks,             # [Batch, Max_Sub]
            key_mask = pathway_masks             # [Batch, Pathway_num, Max_Gene]
        )

        sub2gene_out = self.substructure_embedding_layer(sub2gene_out) # [Batch, Max_Sub, Cross_Attn_dim]

        final_pathway_embedding = torch.amax(gene2sub_out, dim=(1, 2))  # [Batch, Cross_Attn_dim]
        final_drug_embedding, _  = sub2gene_out.max(dim=1)  # [Batch, Cross_Attn_dim]

        # Concatenate and Predict
        combined_embedding = torch.cat((final_pathway_embedding, final_drug_embedding), dim=-1)  # [Batch, pathway_dim + drug_dim]

        x = self.fc1(combined_embedding)
        x = self.bn_fc1(x)  # BatchNorm 적용
        x = F.relu(x)
        x = self.fc2(x)  # [Batch, output_dim]
        
        return x, gene2sub_weights, sub2gene_weights, final_pathway_embedding, final_drug_embedding