import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.embedding_layer import GeneEmbeddingLayer, SubstructureEmbeddingLayer
from modules.diff_cross_attn import Gene2SubDifferCrossMHA, Sub2GeneDifferCrossMHA

#  DRUG RESPONSE MODEL
class DrugResponseModel(nn.Module):
    def __init__(self, pathway_masks, pathway_laplacian_embeddings,
                 gene_layer_dim, substructure_layer_dim, 
                 cross_attn_dim, final_dim,
                 max_gene_slots, max_drug_substructures):
        super(DrugResponseModel, self).__init__()
        self.raw_pathway_masks = pathway_masks # [Pathway_num, Max_Gene_Slots]

        # Embedding Layer (Value)
        self.gene_embedding_layer = GeneEmbeddingLayer(gene_layer_dim)
        self.substructure_embedding_layer = SubstructureEmbeddingLayer(substructure_layer_dim)

        # Positional Embedding Layer (Index 1, ... Max)
        self.max_gene_slots = max_gene_slots
        self.max_drug_substructures = max_drug_substructures
        self.gene_pos_embedding_layer = nn.Embedding(self.max_gene_slots, gene_layer_dim)
        self.drug_pos_embedding_layer = nn.Embedding(self.max_drug_substructures, substructure_layer_dim)
        
        # Spectral Embedding Layer (Laplacian Embedding)
        self.pathway_spectral_embeddings = pathway_laplacian_embeddings  # [Pathway_num, Max_Gene_Slots, laplacian_dim (4)]
        self.pathway_spectral_layer = nn.Linear(4, gene_layer_dim)
        self.drug_spectral_layer = nn.Linear(4, substructure_layer_dim)

        # Cross Attention Layer
        self.Gene2Sub_cross_attention = Gene2SubDifferCrossMHA(gene_embed_dim=gene_layer_dim, sub_embed_dim=substructure_layer_dim, attention_dim=cross_attn_dim, num_heads=4, depth=2)
        self.Sub2Gene_cross_attention = Sub2GeneDifferCrossMHA(sub_embed_dim=substructure_layer_dim, gene_embed_dim=gene_layer_dim, attention_dim=cross_attn_dim, num_heads=4, depth=2)
        
        # Normalization for Each Embedding
        self.gene_value_norm = nn.LayerNorm(gene_layer_dim)
        self.gene_pos_norm = nn.LayerNorm(gene_layer_dim)
        self.gene_spec_norm = nn.LayerNorm(gene_layer_dim)

        self.sub_value_norm = nn.LayerNorm(substructure_layer_dim)
        self.sub_pos_norm = nn.LayerNorm(substructure_layer_dim)
        self.sub_spec_norm = nn.LayerNorm(substructure_layer_dim)
        
        # MLP Layer
        self.fc1 = nn.Linear(2 * cross_attn_dim, final_dim)
        self.bn1 = nn.BatchNorm1d(final_dim)
        self.fc2 = nn.Linear(final_dim, final_dim // 2)
        self.bn2 = nn.BatchNorm1d(final_dim // 2)
        self.fc3 = nn.Linear(final_dim // 2, 1)

    def forward(self, gene_embeddings_input, drug_embeddings_input, drug_spectral_embeddings, drug_masks_input, batch_idx_for_debug=None, current_epoch_for_debug=None): # 디버깅 인자 유지

        current_device = gene_embeddings_input.device
        batch_size = gene_embeddings_input.size(0)

        # Mask
        pathway_masks_for_batch = self.raw_pathway_masks.to(current_device).unsqueeze(0).expand(batch_size, -1, -1)  # [Num_Pathways, Max_Gene_Slots] -> [B, Num_Pathways, Max_Gene_Slots]
        drug_masks_for_attention = drug_masks_input # [B, Max_Drug_Substructures]

        # Embedding Layer (Value)
        gene_embedded_value = self.gene_embedding_layer(gene_embeddings_input) # [B, Num_Pathways, Max_Gene_Slots] -> [B, Num_Pathways, Max_Gene_Slots, gene_layer_dim]
        drug_embedded_value = self.substructure_embedding_layer(drug_embeddings_input) # [B, Max_Drug_Substructures, 768] -> [B, Max_Drug_Substructures, substructure_layer_dim]
        
        # Positional Embedding Layer (Index 1, ... Max)
        gene_pos_embed_base = self.gene_pos_embedding_layer(torch.arange(self.max_gene_slots, device=current_device)) # [Max_Gene_Slots] -> [Max_Gene_Slots, gene_layer_dim]
        gene_pos_embed = gene_pos_embed_base.unsqueeze(0).unsqueeze(0)  # [1, 1, Max_Gene_Slots, gene_layer_dim] 
        drug_pos_embed_base = self.drug_pos_embedding_layer(torch.arange(self.max_drug_substructures, device=current_device)) # [Max_Drug_Substructures] -> [Max_Drug_Substructures, substructure_layer_dim]
        drug_pos_embed = drug_pos_embed_base.unsqueeze(0) # [1, Max_Drug_Substructures, substructure_layer_dim]
        
        # Spectral Embedding Layer (Laplacian Embedding)
        pathway_spec_embed = self.pathway_spectral_layer(self.pathway_spectral_embeddings.to(current_device)) # [Pathway_num, Max_Gene_Slots, gene_layer_dim]
        pathway_spec_embed = pathway_spec_embed.unsqueeze(0) # [1, Pathway_num, Max_Gene_Slots, gene_layer_dim]
        drug_spec_embed = self.drug_spectral_layer(drug_spectral_embeddings) # [B, Max_Drug_Substructures, substructure_layer_dim]
        
        # Summation of Embedding
        gene_embeddings= (
            self.gene_value_norm(gene_embedded_value) +
            self.gene_pos_norm(gene_pos_embed) +
            self.gene_spec_norm(pathway_spec_embed)
        )  # [B, P, G, D]
        
        drug_embeddings = (
            self.sub_value_norm(drug_embedded_value) +
            self.sub_pos_norm(drug_pos_embed) +
            self.sub_spec_norm(drug_spec_embed)
        )  # [B, S, D]
        
        # Gene2Sub Cross attention
        gene2sub_out, gene2sub_weights = self.Gene2Sub_cross_attention(
            query=gene_embeddings,    
            key=drug_embeddings,      
            query_mask=pathway_masks_for_batch,
            key_mask=drug_masks_for_attention
        )
        gene2sub_out = gene2sub_out.masked_fill(
            ~pathway_masks_for_batch.unsqueeze(-1),
            0.0
        )

        # Sub2Gene Cross attention
        sub2gene_out, sub2gene_weights = self.Sub2Gene_cross_attention(
            query=drug_embeddings,    
            key=gene_embeddings,      
            query_mask=drug_masks_for_attention,
            key_mask=pathway_masks_for_batch
        )
        sub2gene_out = sub2gene_out.masked_fill(
            ~drug_masks_for_attention.unsqueeze(-1), # device 일치됨
            0.0
        )

        # Aggregation & MLP
        final_pathway_embedding = torch.amax(gene2sub_out, dim=(1, 2))
        final_drug_embedding, _ = sub2gene_out.max(dim=1)
        combined_embedding = torch.cat((final_pathway_embedding, final_drug_embedding), dim=-1)

        x = F.relu(self.bn1(self.fc1(combined_embedding)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        
        return x, gene2sub_weights, sub2gene_weights, final_pathway_embedding, final_drug_embedding