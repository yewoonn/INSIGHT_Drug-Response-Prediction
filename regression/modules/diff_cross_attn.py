import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from modules.rms_norm import RMSNorm
 
# SOFTMAX
def eps_softmax(x, dim, eps=1e-8):
    x_exp = torch.exp(x)
    x_exp_sum = x_exp.sum(dim=dim, keepdim=True) + eps  
    return x_exp / x_exp_sum

def stable_softmax(x, dim, eps=1e-8):
    # dim을 따라 최대값을 빼줌으로써 수치적 안정성 확보
    x_max, _ = torch.max(x, dim=dim, keepdim=True)
    x_exp = torch.exp(x - x_max)
    x_exp_sum = x_exp.sum(dim=dim, keepdim=True) + eps  
    return x_exp / x_exp_sum

#  GENE2SUB DIFFERENTIAL CROSS-ATTENTION
class Gene2SubDifferCrossAttn(nn.Module):
    """
    Gene:        [B, P, G, E]
    Substructure:[B, S, E]
    Output:      [B, P, G, E]
    """
    def __init__(self, gene_embed_dim: int = 32, sub_embed_dim: int = 768, depth: int = 0):
        super().__init__()
        self.gene_embed_dim = gene_embed_dim
        self.sub_embed_dim = sub_embed_dim
        self.gene_head_dim = gene_embed_dim // 2  
        self.sub_head_dim = sub_embed_dim // 2
        self.scaling = self.gene_head_dim ** -0.5

        self.q_proj = nn.Linear(gene_embed_dim, gene_embed_dim, bias=False)
        self.k_proj = nn.Linear(sub_embed_dim, gene_embed_dim, bias=False)
        self.v_proj = nn.Linear(sub_embed_dim, gene_embed_dim, bias=False)
        self.out_proj = nn.Linear(gene_embed_dim, gene_embed_dim, bias=False)

        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.gene_head_dim).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.gene_head_dim).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.gene_head_dim).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.gene_head_dim).normal_(mean=0, std=0.1))

        self.subln = RMSNorm(2 * self.gene_head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, query: torch.Tensor, key: torch.Tensor, query_mask: torch.Tensor, key_mask: torch.Tensor):
        """
        gene:        [B, P, G, E]
        substructure:[B, S, E]
        Returns:
            out:               [B, P, G, E]
            diff_attn_weights: [B, P, G, S]
        """
        B, P, G, E = query.shape  
        S = key.size(1)

        # Flatten gene -> [B, L, E],  L = P*G
        L = P * G
        gene_flat = query.view(B, L, E)

        # Q, K, V
        Q = self.q_proj(gene_flat)      # [B, L, E]
        K = self.k_proj(key)   # [B, S, E]
        V = self.v_proj(key)   # [B, S, E]
 
        # Separate to 2-Slot(Head)
        Q = Q.view(B, L, 2, self.gene_head_dim).transpose(1, 2)  # [B, 2, L, head_dim]
        K = K.view(B, S, 2, self.gene_head_dim).transpose(1, 2)  # [B, 2, S, head_dim]
        V = V.view(B, S, 2 * self.gene_head_dim)                 # [B, S, 2*head_dim]

        # Scaling & Attention Score
        Q = Q * self.scaling

        attn_scores = torch.matmul(Q, K.transpose(-1, -2))  # [B, 2, L, head_dim] x [B, 2, head_dim, S] => [B, 2, L, S]

        if (query_mask is not None) and (key_mask is not None):
            query_mask = query_mask.to(torch.bool)
            key_mask = key_mask.to(torch.bool)

            # QUERY MASKING
            query_mask = query_mask.to(query.device)
            query_mask_flat = query_mask.view(B, L)  # [B, P, G] => [B, L]
            query_extended_mask = query_mask_flat.unsqueeze(1).unsqueeze(-1)  # [B, 1, L, 1]
            attn_scores = attn_scores.masked_fill(~query_extended_mask, -1e20)

            # KEY MASKING
            key_mask = key_mask.to(query.device)
            key_extended_mask = key_mask.unsqueeze(1).unsqueeze(1) # [B, S] => [B, 1, 1, S]
            attn_scores = attn_scores.masked_fill(~key_extended_mask, -1e20)

        # Softmax
        # attn_scores = F.softmax(attn_scores, dim=-1)  # [B, 2, L, S]
        attn_scores = stable_softmax(attn_scores, dim=-1)

        # Calculate Difference Between Two Slots
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1))
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        diff_attn_weights = attn_scores[:, 0] - lambda_full * attn_scores[:, 1]  # [B, 2, L, S] => [B, L, S]

        min_value = diff_attn_weights.min(dim=-1, keepdim=True).values 
        diff_attn_weights = diff_attn_weights - min_value + 1e-5

        diff_attn_weights = diff_attn_weights.masked_fill(~query_extended_mask.squeeze(1), 0.0) # Masking Query
        diff_attn_weights = diff_attn_weights.masked_fill(~key_extended_mask.squeeze(1), 0.0) # Masking Key

        # V Production
        diff_attn_2d = diff_attn_weights             # [B, L, S]
        attn_out_2d = torch.bmm(diff_attn_2d, V)     #  matmul([B, L, S], [B, S, 2*head_dim]) => [B, L, 2*head_dim]
        attn_out_2d = attn_out_2d.unsqueeze(1)       # [B, 1, L, 2*head_dim]
        attn_out_2d = self.subln(attn_out_2d)        # [B, 1, L, 2*head_dim]
        attn_out_2d = attn_out_2d * (1.0 - self.lambda_init)

        # Final Ouput Projection
        attn_out_2d = attn_out_2d.squeeze(1)         # [B, L, 2*head_dim]
        out = self.out_proj(attn_out_2d)             # [B, L, E]

        # Restore Raw Shape
        out = out.view(B, P, G, E)                  # [B, L, E] => [B, P, G, E]
        diff_attn_weights = diff_attn_weights.view(B, P, G, S) # [B, L, S] => [B, P, G, S]

        return out, diff_attn_weights


#  SUB2GENE DIFFERENTIAL CROSS-ATTENTION
class Sub2GeneDifferCrossAttn(nn.Module):
    """
    Substructure: [B, S, E]
    Gene:         [B, P, G, E]
    Output:       [B, S, E]
    """
    def __init__(self, sub_embed_dim: int = 768, gene_embed_dim: int = 32, depth: int = 0):
        super().__init__()
        self.gene_embed_dim = gene_embed_dim
        self.sub_embed_dim = sub_embed_dim
        self.gene_head_dim = gene_embed_dim // 2  
        self.sub_head_dim = sub_embed_dim // 2
        self.scaling = self.sub_head_dim ** -0.5

        self.q_proj = nn.Linear(sub_embed_dim, sub_embed_dim, bias=False)
        self.k_proj = nn.Linear(gene_embed_dim, sub_embed_dim, bias=False)
        self.v_proj = nn.Linear(gene_embed_dim, sub_embed_dim, bias=False)
        self.out_proj = nn.Linear(sub_embed_dim, sub_embed_dim, bias=False)

        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.sub_head_dim).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.sub_head_dim).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.sub_head_dim).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.sub_head_dim).normal_(mean=0, std=0.1))

        self.subln = RMSNorm(2 * self.sub_head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, query: torch.Tensor, key: torch.Tensor, query_mask: torch.Tensor, key_mask: torch.Tensor):
        """
        substructure: [B, S, E]
        gene:         [B, P, G, E]
        Returns:
            out:               [B, S, E]
            diff_attn_weights: [B, S, P, G]
        """

        B, S, E = query.shape
        _, P, G, _ = key.shape
        L = P * G 

        # Projection Q, K, V
        Q = self.q_proj(query)       # [B, S, E]
        K = self.k_proj(key)         # [B, P, G, E]
        V = self.v_proj(key)         # [B, P, G, E]

        # Flatten gene
        K = K.view(B, L, E) # [B, L, E],  L = P*G

        # Separate to 2-Slot(Head)
        Q = Q.view(B, S, 2, self.sub_head_dim).transpose(1, 2)  # [B, S, E] => [B, 2, S, head_dim]
        K = K.view(B, L, 2, self.sub_head_dim).transpose(1, 2)  # [B, L, E] => [B, 2, L, head_dim]
        V = V.view(B, L, 2 * self.sub_head_dim)                 # [B, L, E] => [B, L, 2*head_dim]

        # Scaling + Attention Score
        Q = Q * self.scaling
        attn_scores = torch.matmul(Q, K.transpose(-1, -2))  # [B, 2, S, head_dim] x [B, 2, head_dim, L] => [B, 2, S, L]

        if (query_mask is not None) and (key_mask is not None):
            # QUERY MASKING            
            query_mask = query_mask.to(query.device) # [B, S]
            query_extended_mask = query_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, S, 1]
            attn_scores = attn_scores.masked_fill(~query_extended_mask, -1e20)

            # KEY MASKING
            key_mask = key_mask.to(query.device)  # [B, P, G]
            key_mask_flat = key_mask.view(B, L)   # [B, L]
            key_extended_mask = key_mask_flat.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, L]
            attn_scores = attn_scores.masked_fill(~key_extended_mask, -1e20)

        # Softmax
        attn_scores = stable_softmax(attn_scores, dim=-1) # [B, 2, S, L]

        # Calculate Difference Between Two Slots
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1))
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        diff_attn_weights = attn_scores[:, 0] - lambda_full * attn_scores[:, 1]  # => [B, S, L]

        min_value = diff_attn_weights.min(dim=-1, keepdim=True).values  # Find minimum value along the last dimension
        diff_attn_weights = diff_attn_weights - min_value + 1e-5

        diff_attn_weights = diff_attn_weights.masked_fill(~query_extended_mask.squeeze(1), 0.0) # Query Masking
        diff_attn_weights = diff_attn_weights.masked_fill(~key_extended_mask.squeeze(1), 0.0) # Key Masking

        # V Production
        attn_out_2d = torch.bmm(diff_attn_weights, V)  # [B, S, 2*head_dim]
        attn_out_2d = attn_out_2d.unsqueeze(1)         # [B, 1, S, 2*head_dim]
        attn_out_2d = self.subln(attn_out_2d)          # [B, 1, S, 2*head_dim]
        attn_out_2d = attn_out_2d * (1.0 - self.lambda_init)

        # Final Ouput Projection
        attn_out_2d = attn_out_2d.squeeze(1)           # [B, S, 2*head_dim]
        out = self.out_proj(attn_out_2d)               # [B, S, E]

        # Restore Raw Shape
        diff_attn_weights = diff_attn_weights.view(B, S, P, G)

        return out, diff_attn_weights