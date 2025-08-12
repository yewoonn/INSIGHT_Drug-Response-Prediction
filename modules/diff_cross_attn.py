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

class Path2SubDifferCrossMHA(nn.Module):
    """
    Pathway:  [B, P, E]
    Substrate: [B, S, E]
    Output:   [B, P, E]
    """
    def __init__(self, pathway_embed_dim: int, drug_embed_dim: int, attention_dim: int, num_heads: int, depth: int):
        super().__init__()
        if attention_dim % (2 * num_heads) != 0:
            raise ValueError(
                f"embed_dim ({attention_dim}) must be divisible by 2 * num_heads ({2*num_heads})"
            )
        
        self.pathway_embed_dim = pathway_embed_dim
        self.drug_embed_dim = drug_embed_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // (2 * num_heads)
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(pathway_embed_dim, 2 * num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(drug_embed_dim, 2 * num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(drug_embed_dim, 2 * num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(2 * num_heads * self.head_dim, pathway_embed_dim, bias=False)

        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, query: torch.Tensor, key: torch.Tensor, key_mask: torch.Tensor = None):
        """
        query(pathway): [B, P, E]
        key(drug): [B, S, E]
        output: [B, P, E]
        """
        B, P, E = query.shape
        _, S, E = key.shape
        
        Q = (self.q_proj(query)
             .view(B, P, self.num_heads, 2, self.head_dim)
             .permute(0, 2, 3, 1, 4))
        K = (self.k_proj(key)
             .view(B, S, self.num_heads, 2, self.head_dim)
             .permute(0, 2, 3, 1, 4))
        V = (self.v_proj(key)
             .view(B, S, self.num_heads, 2, self.head_dim)
             .permute(0, 2, 3, 1, 4))

        Q = Q * self.scaling
        scores = torch.matmul(Q, K.transpose(-1, -2))
        if key_mask is not None:
            # key_mask: [B, S] (True = valid, False = padding)
            key_mask = key_mask.to(Q.device)
            key_mask_exp = key_mask.unsqueeze(1).unsqueeze(2).unsqueeze(1)  # [B, 1, 1, 1, S]
            key_mask_exp = key_mask_exp.expand(B, self.num_heads, 2, P, S)
            scores = scores.masked_fill(~key_mask_exp, -1e20)

        scores = stable_softmax(scores, dim=-1)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(Q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(Q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        
        slot0 = scores[:, :, 0]
        slot1 = scores[:, :, 1]
        diff_attn = slot0 - lambda_full * slot1

        V_cat = (V.permute(0,1,3,2,4)
                 .reshape(B, self.num_heads, S, 2*self.head_dim))

        out = torch.matmul(diff_attn, V_cat)
        out = self.subln(out) * (1 - self.lambda_init)
        out = (out.permute(0,2,1,3)
               .reshape(B, P, 2*self.num_heads*self.head_dim))
        
        out = self.out_proj(out).view(B, P, E)
        diff_attn = diff_attn.view(B, self.num_heads, P, S)

        return out, diff_attn
    
class Drug2PathDifferCrossMHA(nn.Module):
    """
    Drug:     [B, 1, E]
    Pathway:  [B, P, E]
    Output:   [B, 1, E]
    """
    def __init__(self, drug_embed_dim: int, pathway_embed_dim: int, attention_dim: int, num_heads: int, depth: int):
        super().__init__()
        if attention_dim % (2 * num_heads) != 0:
            raise ValueError(
                f"embed_dim ({attention_dim}) must be divisible by 2 * num_heads ({2*num_heads})"
            )

        self.drug_embed_dim = drug_embed_dim
        self.pathway_embed_dim = pathway_embed_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // (2 * num_heads)
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(drug_embed_dim, 2 * num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(pathway_embed_dim, 2 * num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(pathway_embed_dim, 2 * num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(2 * num_heads * self.head_dim, drug_embed_dim, bias=False)

        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, query: torch.Tensor, key: torch.Tensor, key_mask: torch.Tensor = None):
        """
        query(drug):   [B, 1, E]
        key(pathway):  [B, P, E]
        Returns:
            out:        [B, 1, E]
            diff_attn:  [B, H, 1, P]
        """
        B, _, _ = query.shape
        _, P, E = key.shape

        Q = self.q_proj(query).view(B, 1, self.num_heads, 2, self.head_dim).permute(0, 2, 3, 1, 4)
        K = self.k_proj(key).view(B, P, self.num_heads, 2, self.head_dim).permute(0, 2, 3, 1, 4)
        V = self.v_proj(key).view(B, P, self.num_heads, 2, self.head_dim).permute(0, 2, 3, 1, 4)

        Q = Q * self.scaling
        scores = torch.matmul(Q, K.transpose(-1, -2))  # [B, H, 2, 1, P]
        scores = stable_softmax(scores, dim=-1)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1)).type_as(Q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2)).type_as(Q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        slot0 = scores[:, :, 0]  # [B, H, 1, P]
        slot1 = scores[:, :, 1]  # [B, H, 1, P]
        diff_attn = slot0 - lambda_full * slot1  # [B, H, 1, P]

        V_cat = V.permute(0, 1, 3, 2, 4).reshape(B, self.num_heads, P, 2 * self.head_dim)
        out = torch.matmul(diff_attn, V_cat)  # [B, H, 1, 2D]
        out = self.subln(out) * (1 - self.lambda_init)
        out = out.permute(0, 2, 1, 3).reshape(B, 1, 2 * self.num_heads * self.head_dim)
        out = self.out_proj(out)  # [B, 1, E]

        return out, diff_attn
