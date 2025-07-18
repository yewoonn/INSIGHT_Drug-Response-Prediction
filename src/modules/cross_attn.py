import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from modules.rms_norm import RMSNorm

def stable_softmax(x, dim, eps=1e-20):
    x = x.to(dtype=torch.float32)
    x_max, _ = torch.max(x, dim=dim, keepdim=True)
    x_exp = torch.exp(x - x_max)
    x_exp_sum = x_exp.sum(dim=dim, keepdim=True) + eps
    return x_exp / x_exp_sum

# Vanilla Cross-Attention
class Path2DrugCrossMHA(nn.Module):
    """
    Pathway:  [B, P, E]
    Drug:     [B, E]
    Output:   [B, P, E]
    """
    def __init__(self, pathway_embed_dim: int, drug_embed_dim: int, attention_dim: int, num_heads: int, depth: int):
        super().__init__()
        if attention_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({attention_dim}) must be divisible by num_heads ({num_heads})"
            )
        
        self.pathway_embed_dim = pathway_embed_dim
        self.drug_embed_dim = drug_embed_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(pathway_embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(drug_embed_dim, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(drug_embed_dim, num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, pathway_embed_dim, bias=False)

        self.norm = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, query: torch.Tensor, key: torch.Tensor, query_mask: torch.Tensor = None, key_mask: torch.Tensor = None):
        """
        query(pathway): [B, P, E]
        key(drug):      [B, E]
        Returns:
            out:               [B, P, E]
            attn_weights:      [B, H, P, 1]
        """
        B, P, E = query.shape
        _, drug_dim = key.shape
        
        # Drug key를 [B, 1, E]로 unsqueeze
        key = key.unsqueeze(1)  # [B, 1, E]
        
        Q = (self.q_proj(query)
             .view(B, P, self.num_heads, self.head_dim)
             .permute(0, 2, 1, 3))  # [B, H, P, head_dim]
        K = (self.k_proj(key)
             .view(B, 1, self.num_heads, self.head_dim)
             .permute(0, 2, 1, 3))  # [B, H, 1, head_dim]
        V = (self.v_proj(key)
             .view(B, 1, self.num_heads, self.head_dim)
             .permute(0, 2, 1, 3))  # [B, H, 1, head_dim]

        Q = Q * self.scaling
        scores = torch.matmul(Q, K.transpose(-1, -2))  # [B, H, P, 1]

        # 마스킹 (선택적)
        if query_mask is not None:
            query_mask = query_mask.to(Q.device)  # [B, P]
            query_mask_exp = query_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, P, 1]
            query_mask_exp = query_mask_exp.expand(B, self.num_heads, P, 1)
            scores = scores.masked_fill(~query_mask_exp, -1e20)

        attn_weights = stable_softmax(scores, dim=-1)  # [B, H, P, 1]
        
        out = torch.matmul(attn_weights, V)  # [B, H, P, head_dim]
        out = self.norm(out)
        out = (out.permute(0, 2, 1, 3)
               .reshape(B, P, self.num_heads * self.head_dim))  # [B, P, attention_dim]
        
        out = self.out_proj(out).view(B, P, E)
        attn_weights = attn_weights.view(B, self.num_heads, P, 1)

        return out, attn_weights

class Drug2PathCrossMHA(nn.Module):
    """
    Drug:     [B, E]
    Pathway:  [B, P, E]
    Output:   [B, E]
    """
    def __init__(self, drug_embed_dim: int, pathway_embed_dim: int, attention_dim: int, num_heads: int, depth: int):
        super().__init__()
        if attention_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({attention_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.drug_embed_dim = drug_embed_dim
        self.pathway_embed_dim = pathway_embed_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(drug_embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(pathway_embed_dim, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(pathway_embed_dim, num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, drug_embed_dim, bias=False)

        self.norm = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, query: torch.Tensor, key: torch.Tensor, query_mask: torch.Tensor = None, key_mask: torch.Tensor = None):
        """
        query(drug):     [B, E]
        key(pathway):    [B, P, E]
        Returns:
            out:               [B, E]
            attn_weights:      [B, H, 1, P]
        """
        B, drug_dim = query.shape
        _, P, pathway_dim = key.shape
        
        # Drug query를 [B, 1, E]로 unsqueeze
        query = query.unsqueeze(1)  # [B, 1, E]

        Q = (self.q_proj(query)
             .view(B, 1, self.num_heads, self.head_dim)
             .permute(0, 2, 1, 3))  # [B, H, 1, head_dim]
        K = (self.k_proj(key)
             .view(B, P, self.num_heads, self.head_dim)
             .permute(0, 2, 1, 3))  # [B, H, P, head_dim]
        V = (self.v_proj(key)
             .view(B, P, self.num_heads, self.head_dim)
             .permute(0, 2, 1, 3))  # [B, H, P, head_dim]

        Q = Q * self.scaling
        scores = torch.matmul(Q, K.transpose(-1, -2))  # [B, H, 1, P]

        # 마스킹 (선택적)
        if key_mask is not None:
            key_mask = key_mask.to(Q.device)  # [B, P]
            key_mask_exp = key_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, P]
            key_mask_exp = key_mask_exp.expand(B, self.num_heads, 1, P)
            scores = scores.masked_fill(~key_mask_exp, -1e20)

        attn_weights = stable_softmax(scores, dim=-1)  # [B, H, 1, P]

        out = torch.matmul(attn_weights, V)  # [B, H, 1, head_dim]
        out = self.norm(out)
        out = (out.permute(0, 2, 1, 3)
               .reshape(B, 1, self.num_heads * self.head_dim))  # [B, 1, attention_dim]
        out = self.out_proj(out).squeeze(1)  # [B, E]

        attn_weights = attn_weights.view(B, self.num_heads, 1, P)
        return out, attn_weights