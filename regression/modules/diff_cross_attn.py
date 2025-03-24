import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from modules.rms_norm import RMSNorm

def stable_softmax(x, dim, eps=1e-8):
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
        self.scaling = self.gene_head_dim ** -0.5

        self.q_proj = nn.Linear(gene_embed_dim, gene_embed_dim, bias=False)
        self.kv_proj = nn.Linear(sub_embed_dim, 2 * gene_embed_dim, bias=False)
        self.out_proj = nn.Linear(gene_embed_dim, gene_embed_dim, bias=False)

        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.gene_head_dim).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.gene_head_dim).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.gene_head_dim).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.gene_head_dim).normal_(mean=0, std=0.1))

        self.subln = RMSNorm(2 * self.gene_head_dim, eps=1e-5, elementwise_affine=True)

    def _compute_lambda(self):
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1))
        return lambda_1 - lambda_2 + self.lambda_init

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
        L = P * G

        gene_flat = query.view(B, L, E)

        Q = self.q_proj(gene_flat)      # [B, L, E]
        KV = self.kv_proj(key)          # [B, S, 2*gene_embed_dim]
        K, V = KV.split(self.gene_embed_dim, dim=-1)

        # 2-slot head로 재구성
        Q = Q.view(B, L, 2, self.gene_head_dim).permute(0, 2, 1, 3)  # [B, 2, L, gene_head_dim]
        K = K.view(B, S, 2, self.gene_head_dim).permute(0, 2, 1, 3)  # [B, 2, S, gene_head_dim]
        V = V.view(B, S, 2 * self.gene_head_dim).unsqueeze(1)         # [B, 1, S, 2*gene_head_dim]

        Q = Q * self.scaling

        # Masking
        if (query_mask is not None) and (key_mask is not None):
            query_mask = query_mask.to(Q.device)
            key_mask = key_mask.to(Q.device)
            query_mask_flat = query_mask.view(B, L)  # [B, L]
            combined_mask = query_mask_flat.unsqueeze(-1) & key_mask.unsqueeze(1)  # [B, L, S]
            combined_mask_expanded = combined_mask.unsqueeze(1)  # [B, 1, L, S]
            attn_scores = torch.matmul(Q, K.transpose(-1, -2))  # [B, 2, L, S]
            attn_scores = attn_scores.masked_fill(~combined_mask_expanded, -1e20)

        attn_scores = stable_softmax(attn_scores, dim=-1)

        lambda_full = self._compute_lambda()

        diff_attn_weights = attn_scores[:, 0] - lambda_full * attn_scores[:, 1]  # [B, L, S]

        min_value = diff_attn_weights.min(dim=-1, keepdim=True).values 
        diff_attn_weights.sub_(min_value).add_(1e-20)

        if (query_mask is not None) and (key_mask is not None):
            diff_attn_weights = diff_attn_weights.masked_fill(~combined_mask, 0.0)

        attn_out_2d = torch.bmm(diff_attn_weights, V.squeeze(1))  # [B, L, 2*gene_head_dim]
        attn_out_2d = attn_out_2d.unsqueeze(1)  # [B, 1, L, 2*gene_head_dim]
        attn_out_2d = self.subln(attn_out_2d)
        attn_out_2d.mul_(1.0 - self.lambda_init)

        attn_out_2d = attn_out_2d.squeeze(1)  # [B, L, 2*gene_head_dim]
        out = self.out_proj(attn_out_2d)      # [B, L, gene_embed_dim]
        out = out.view(B, P, G, self.gene_embed_dim)
        diff_attn_weights = diff_attn_weights.view(B, P, G, S)

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
        self.sub_head_dim = sub_embed_dim // 2
        self.scaling = self.sub_head_dim ** -0.5

        self.q_proj = nn.Linear(sub_embed_dim, sub_embed_dim, bias=False)
        self.kv_proj = nn.Linear(gene_embed_dim, 2 * sub_embed_dim, bias=False)
        self.out_proj = nn.Linear(sub_embed_dim, sub_embed_dim, bias=False)

        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.sub_head_dim).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.sub_head_dim).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.sub_head_dim).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.sub_head_dim).normal_(mean=0, std=0.1))

        self.subln = RMSNorm(2 * self.sub_head_dim, eps=1e-5, elementwise_affine=True)

    def _compute_lambda(self):
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1))
        return lambda_1 - lambda_2 + self.lambda_init

    def forward(self, query: torch.Tensor, key: torch.Tensor, query_mask: torch.Tensor, key_mask: torch.Tensor):
        """
        substructure: [B, S, E]
        gene:         [B, P, G, E]
        Returns:
            out:               [B, S, E]
            diff_attn_weights: [B, S, P, G]
        """
        B, S, _ = query.shape
        _, P, G, _ = key.shape
        L = P * G 

        # Projection
        Q = self.q_proj(query)  # [B, S, sub_embed_dim]
        gene_flat = key.view(B, L, self.gene_embed_dim)  # [B, L, gene_embed_dim]
        KV = self.kv_proj(gene_flat)  # [B, L, 2*sub_embed_dim]
        K, V = KV.split(self.sub_embed_dim, dim=-1)

        # 2-slot head로 재구성
        Q = Q.view(B, S, 2, self.sub_head_dim).permute(0, 2, 1, 3)  # [B, 2, S, sub_head_dim]
        K = K.view(B, L, 2, self.sub_head_dim).permute(0, 2, 1, 3)  # [B, 2, L, sub_head_dim]
        V = V.view(B, L, 2 * self.sub_head_dim).unsqueeze(1)         # [B, 1, L, 2*sub_head_dim]

        Q = Q * self.scaling

        # Masking
        if (query_mask is not None) and (key_mask is not None):
            query_mask = query_mask.to(Q.device)
            key_mask = key_mask.to(Q.device)
            key_mask_flat = key_mask.view(B, L)
            combined_mask = query_mask.unsqueeze(-1) & key_mask_flat.unsqueeze(1)  # [B, S, L]
            combined_mask_expanded = combined_mask.unsqueeze(1)  # [B, 1, S, L]
            attn_scores = torch.matmul(Q, K.transpose(-1, -2))  # [B, 2, S, L]
            attn_scores = attn_scores.masked_fill(~combined_mask_expanded, -1e20)

        attn_scores = stable_softmax(attn_scores, dim=-1)

        lambda_full = self._compute_lambda()

        diff_attn_weights = attn_scores[:, 0] - lambda_full * attn_scores[:, 1]  # [B, S, L]

        min_value = diff_attn_weights.min(dim=-1, keepdim=True).values
        diff_attn_weights.sub_(min_value).add_(1e-20)

        if (query_mask is not None) and (key_mask is not None):
            diff_attn_weights = diff_attn_weights.masked_fill(~combined_mask, 0.0)

        attn_out_2d = torch.bmm(diff_attn_weights, V.squeeze(1))  # [B, S, 2*sub_head_dim]
        attn_out_2d = attn_out_2d.unsqueeze(1)  # [B, 1, S, 2*sub_head_dim]
        attn_out_2d = self.subln(attn_out_2d)
        attn_out_2d.mul_(1.0 - self.lambda_init)

        attn_out_2d = attn_out_2d.squeeze(1)  # [B, S, 2*sub_head_dim]
        out = self.out_proj(attn_out_2d)      # [B, S, sub_embed_dim]

        diff_attn_weights = diff_attn_weights.view(B, S, P, G)

        return out, diff_attn_weights
