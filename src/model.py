import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.diff_cross_attn import Path2DrugDifferCrossMHA, Drug2PathDifferCrossMHA
from modules.cross_attn import Path2DrugCrossMHA, Drug2PathCrossMHA

#  1. FFN 모듈 (Gene)
class CellLineGeneFFN(nn.Module):
    """(B, Max_Genes, 10) 입력을 받아 (B, output_dim)을 출력하는 FFN"""
    def __init__(self, max_genes, output_dim, input_dim=10, hidden_dim=1024, dropout_rate=0.5):
        super(CellLineGeneFFN, self).__init__()
        self.flattened_input_dim = max_genes * input_dim
        self.network = nn.Sequential(
            nn.Linear(self.flattened_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        B = x.shape[0]
        x_flat = x.view(B, -1)
        return self.network(x_flat)

#  2. FFN 모듈 (Drug) - ChemBERTa embeddings 처리
class DrugFFN(nn.Module):
    """(B, 768) ChemBERTa 입력을 받아 (B, output_dim)을 출력하는 FFN"""
    def __init__(self, input_dim=768, output_dim=64, hidden_dim=1024, dropout_rate=0.5):
        super(DrugFFN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


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
        self.gene_ffn = CellLineGeneFFN(max_genes=max_gene_slots, input_dim=gene_input_dim, output_dim=gene_ffn_output_dim, 
                                       hidden_dim=gene_ffn_hidden_dim, dropout_rate=gene_ffn_dropout)
        self.drug_ffn = DrugFFN(input_dim=drug_input_dim, output_dim=drug_ffn_output_dim,
                               hidden_dim=drug_ffn_hidden_dim, dropout_rate=drug_ffn_dropout)
        
        # Cross-Attention Modules Initialization
        if isDiffer:
            self.Path2Drug_cross_attention = Path2DrugDifferCrossMHA(
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
            self.Path2Drug_cross_attention = Path2DrugCrossMHA(
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
        self.fc1 = nn.Linear(2 * cross_attn_dim, final_dim)
        self.bn1 = nn.BatchNorm1d(final_dim)
        self.fc2 = nn.Linear(final_dim, final_dim // final_dim_reduction_factor)
        self.bn2 = nn.BatchNorm1d(final_dim // final_dim_reduction_factor)
        self.fc3 = nn.Linear(final_dim // final_dim_reduction_factor, 1)

    def forward(self, gene_embeddings_input, drug_chembert_embeddings):
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
        pathway_embeddings = self.gene_value_norm(gene_embedded_value)  # [B, P, gene_ffn_output_dim]

        # Drug FFN Module
        drug_embeddings = self.drug_ffn(drug_chembert_embeddings)  # [B, drug_ffn_output_dim]
        
        # Path2Drug Cross-Attention
        path2drug_out, path2drug_weights = self.Path2Drug_cross_attention(
            query=pathway_embeddings,    # [B, P, gene_ffn_output_dim]
            key=drug_embeddings,         # [B, drug_ffn_output_dim]
        )

        # Drug2Path Cross-Attention  
        drug2path_out, drug2path_weights = self.Drug2Path_cross_attention(
            query=drug_embeddings,       # [B, drug_ffn_output_dim]
            key=pathway_embeddings,      # [B, P, gene_ffn_output_dim]
        )

        # Pooling operations
        final_pathway_embedding, _ = path2drug_out.max(dim=1)  # [B, gene_ffn_output_dim]
        final_drug_embedding = drug2path_out                   # [B, drug_ffn_output_dim]
        
        combined_embedding = torch.cat((final_pathway_embedding, final_drug_embedding), dim=-1)

        x = self.dropout(F.relu(self.bn1(self.fc1(combined_embedding))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x, path2drug_weights, drug2path_weights, gene_embedded_value, drug_embeddings, pathway_embeddings