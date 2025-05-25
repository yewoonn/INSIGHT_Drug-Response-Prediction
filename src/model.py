import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.embedding_layer import GeneEmbeddingLayer, SubstructureEmbeddingLayer
from modules.diff_cross_attn import Gene2SubDifferCrossMHA, Sub2GeneDifferCrossMHA

#  DRUG RESPONSE MODEL
class DrugResponseModel(nn.Module):
    def __init__(self, pathway_masks, 
                 gene_layer_dim, substructure_layer_dim, 
                 cross_attn_dim, final_dim,
                 max_gene_slots, max_drug_substructures):
        super(DrugResponseModel, self).__init__()
        
        # 초기 설정
        self.raw_pathway_masks = pathway_masks # [Pathway_num, Max_Gene_Slots]
        self.max_gene_slots = max_gene_slots
        self.max_drug_substructures = max_drug_substructures
        
        # Value 기반 임베딩 레이어
        self.gene_embedding_layer = GeneEmbeddingLayer(gene_layer_dim)
        self.substructure_embedding_layer = SubstructureEmbeddingLayer(substructure_layer_dim)

        # Position 기반 임베딩 레이어
        self.gene_pos_embedding_layer = nn.Embedding(self.max_gene_slots, gene_layer_dim)
        self.drug_pos_embedding_layer = nn.Embedding(self.max_drug_substructures, substructure_layer_dim)

        # 크로스 어텐션 레이어
        self.Gene2Sub_cross_attention = Gene2SubDifferCrossMHA(gene_embed_dim=gene_layer_dim, sub_embed_dim=substructure_layer_dim, attention_dim=cross_attn_dim, num_heads=4, depth=2)
        self.Sub2Gene_cross_attention = Sub2GeneDifferCrossMHA(sub_embed_dim=substructure_layer_dim, gene_embed_dim=gene_layer_dim, attention_dim=cross_attn_dim, num_heads=4, depth=2)
        
        # MLP 레이어
        self.fc1 = nn.Linear(2 * cross_attn_dim, final_dim)
        self.bn1 = nn.BatchNorm1d(final_dim)
        self.fc2 = nn.Linear(final_dim, final_dim // 2)
        self.bn2 = nn.BatchNorm1d(final_dim // 2)
        self.fc3 = nn.Linear(final_dim // 2, 1)

    def forward(self, gene_embeddings_input, drug_embeddings_input, drug_masks_input, batch_idx_for_debug=None, current_epoch_for_debug=None): 
        current_device = gene_embeddings_input.device
        batch_size = gene_embeddings_input.size(0)

        # 패스웨이 마스크
        # self.raw_pathway_masks: [Num_Pathways, Max_Gene_Slots] >> pathway_masks_for_batch: [B, Num_Pathways, Max_Gene_Slots]
        pathway_masks_for_batch = self.raw_pathway_masks.to(current_device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # 약물 마스크
        drug_masks_for_attention = drug_masks_input   # [B, Max_Drug_Substructures]

        # Value 기반 임베딩
        gene_embedded_value = self.gene_embedding_layer(gene_embeddings_input) # [B, Num_Pathways, Max_Gene_Slots, gene_layer_dim]
        drug_embedded_value = self.substructure_embedding_layer(drug_embeddings_input) # [B, Max_Drug_Substructures, substructure_layer_dim]
        
        # Position 기반 임베딩
        gene_position_ids = torch.arange(self.max_gene_slots, device=current_device) # [Max_Gene_Slots]
        gene_pos_embed_base = self.gene_pos_embedding_layer(gene_position_ids) # [Max_Gene_Slots, gene_layer_dim]
        gene_pos_embed = gene_pos_embed_base.unsqueeze(0).unsqueeze(0) # [1, 1, Max_Gene_Slots, gene_layer_dim]
        
        drug_position_ids = torch.arange(self.max_drug_substructures, device=current_device) # [Max_Drug_Substructures]
        drug_pos_embed_base = self.drug_pos_embedding_layer(drug_position_ids) # [Max_Drug_Substructures, substructure_layer_dim]   
        drug_pos_embed = drug_pos_embed_base.unsqueeze(0) # [1, Max_Drug_Substructures, substructure_layer_dim]
        
        # Value + Position 임베딩
        gene_embeddings_with_pos = gene_embedded_value + gene_pos_embed
        drug_embeddings_with_pos = drug_embedded_value + drug_pos_embed
       
        # 크로스 어텐션 레이어
        # 1) Gene2Sub
        gene2sub_out, gene2sub_weights = self.Gene2Sub_cross_attention(
            query=gene_embeddings_with_pos,    
            key=drug_embeddings_with_pos,      
            query_mask=pathway_masks_for_batch,
            key_mask=drug_masks_for_attention
        )
        gene2sub_out = gene2sub_out.masked_fill(
            ~pathway_masks_for_batch.unsqueeze(-1),
            0.0
        )

        # 2) Sub2Gene
        sub2gene_out, sub2gene_weights = self.Sub2Gene_cross_attention(
            query=drug_embeddings_with_pos,    
            key=gene_embeddings_with_pos,      
            query_mask=drug_masks_for_attention,
            key_mask=pathway_masks_for_batch
        )
        sub2gene_out = sub2gene_out.masked_fill(
            ~drug_masks_for_attention.unsqueeze(-1), 
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