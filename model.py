import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.diff_cross_attn import Path2SubDifferCrossMHA, Drug2PathDifferCrossMHA
from modules.cross_attn import Path2SubCrossMHA, Drug2PathCrossMHA
from modules.ffn_layer import CelllineFFN, DrugFFN
    
#  DRUG RESPONSE MODEL
class DrugResponseModel(nn.Module):
    def __init__(self, pathway_gene_indices,
                 gene_ffn_output_dim, drug_ffn_output_dim, 
                 cross_attn_dim, final_dim,
                 max_gene_slots, gene_input_dim=10, drug_input_dim=768, isDiffer=True,
                 gene_ffn_hidden_dim=1024, drug_ffn_hidden_dim=1024,
                 gene_ffn_dropout=0.5, drug_ffn_dropout=0.5,
                 num_heads=4, depth=2, mlp_dropout=0.3, final_dim_reduction_factor=2):
        super(DrugResponseModel, self).__init__()
        
        self.register_buffer('pathway_gene_indices', pathway_gene_indices)
        self.max_gene_slots = max_gene_slots
        self.gene_value_norm = nn.LayerNorm(gene_ffn_output_dim)

        # FFN Modules Initialization
        self.gene_ffn = CelllineFFN(max_genes=max_gene_slots, input_dim=gene_input_dim, output_dim=gene_ffn_output_dim, 
                                       hidden_dim=gene_ffn_hidden_dim, dropout_rate=gene_ffn_dropout)
        self.drug_ffn = DrugFFN(input_dim=drug_input_dim, output_dim=drug_ffn_output_dim,
                                         hidden_dim=drug_ffn_hidden_dim, dropout_rate=drug_ffn_dropout)
        self.sub_ffn = DrugFFN(input_dim=drug_input_dim, output_dim=drug_ffn_output_dim,
                                         hidden_dim=drug_ffn_hidden_dim, dropout_rate=drug_ffn_dropout)
                
        # Cross-Attention Modules Initialization
        if isDiffer:
            self.Path2Drug_cross_attention = Path2SubDifferCrossMHA(
                pathway_embed_dim=gene_ffn_output_dim, 
                drug_embed_dim=drug_ffn_output_dim, 
                attention_dim=cross_attn_dim, 
                num_heads=num_heads, 
                depth=depth
            )
            self.Drug2Path_cross_attention = Drug2PathDifferCrossMHA(
                drug_embed_dim=drug_ffn_output_dim, 
                pathway_embed_dim=gene_ffn_output_dim, 
                attention_dim=cross_attn_dim, 
                num_heads=num_heads, 
                depth=depth
            )
        else:
            self.Path2Drug_cross_attention = Path2SubCrossMHA(
                pathway_embed_dim=gene_ffn_output_dim, 
                drug_embed_dim=drug_ffn_output_dim, 
                attention_dim=cross_attn_dim, 
                num_heads=num_heads, 
                depth=depth
            )
            self.Drug2Path_cross_attention = Drug2PathCrossMHA(
                drug_embed_dim=drug_ffn_output_dim, 
                pathway_embed_dim=gene_ffn_output_dim, 
                attention_dim=cross_attn_dim, 
                num_heads=num_heads, 
                depth=depth
            )

        self.dropout = nn.Dropout(mlp_dropout)
        self.fc1 = nn.Linear(3 * cross_attn_dim, final_dim)
        self.bn1 = nn.BatchNorm1d(final_dim)
        self.fc2 = nn.Linear(final_dim, final_dim // final_dim_reduction_factor)
        self.bn2 = nn.BatchNorm1d(final_dim // final_dim_reduction_factor)
        self.fc3 = nn.Linear(final_dim // final_dim_reduction_factor, 1)

    def forward(self, gene_embeddings_input, drug_embeddings_input, drug_substructure_embeddings, drug_multitoken_masks):
        current_device = gene_embeddings_input.device
        batch_size = gene_embeddings_input.size(0)
        num_pathways = self.pathway_gene_indices.shape[0]

        # Pathway Instantiation
        indices = self.pathway_gene_indices.clone()
        indices[indices == -1] = 0 
        expanded_genes = gene_embeddings_input.unsqueeze(1).expand(-1, num_pathways, -1, -1)
        expanded_indices = indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, 10)
        pathway_specific_genes = torch.gather(expanded_genes, 2, expanded_indices.to(current_device))

        # Gene FFN Module
        B, P, G, D = pathway_specific_genes.shape
        ffn_input_reshaped = pathway_specific_genes.view(B * P, G, D)
        ffn_output = self.gene_ffn(ffn_input_reshaped)
        gene_embedded_value = ffn_output.view(B, P, -1)
        pathway_embeddings = gene_embedded_value
        pathway_embeddings = self.gene_value_norm(gene_embedded_value)  # [B, P, gene_ffn_output_dim]

        # Multi-Token Drug FFN Module
        sub_embeddings = self.sub_ffn(drug_substructure_embeddings, drug_multitoken_masks)  # [B, L, drug_ffn_output_dim]
        drug_embeddings_input = drug_embeddings_input.unsqueeze(1)  # [B, drug_ffn_output_dim] -> [B, 1, drug_ffn_output_dim]
        drug_embeddings = self.drug_ffn(drug_embeddings_input)  # [B, 768] -> [B, 1, drug_ffn_output_dim]

        # Path2Drug Cross-Attention (pathway to drug sequence)
        path2drug_out, path2drug_weights = self.Path2Drug_cross_attention(
            query=pathway_embeddings,    # [B, P, gene_ffn_output_dim]
            key=sub_embeddings,         # [B, L, drug_ffn_output_dim]
            key_mask=drug_multitoken_masks  # [B, L]
        )

        # Drug2Path Cross-Attention
        drug2path_out, drug2path_weights = self.Drug2Path_cross_attention(
            query=drug_embeddings,       # [B, 1, drug_ffn_output_dim]
            key=pathway_embeddings,      # [B, P, gene_ffn_output_dim]
        )

        # Max pooling for pathway embeddings
        final_pathway_embedding, _ = path2drug_out.max(dim=1)  # [B, gene_ffn_output_dim]
        final_drug_embedding, _ = drug2path_out.max(dim=1)  # [B, drug_ffn_output_dim]
        final_sub_embedding, _ = sub_embeddings.max(dim=1)  # [B, drug_ffn_output_dim]

        # Concatenate embeddings and MLP
        combined_embedding = torch.cat((final_pathway_embedding, final_drug_embedding, final_sub_embedding), dim=-1)
        x = self.dropout(F.relu(self.bn1(self.fc1(combined_embedding))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x, path2drug_weights, drug2path_weights