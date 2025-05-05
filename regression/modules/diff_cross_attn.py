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


#  GENE2SUB DIFFERENTIAL CROSS-ATTENTION
class Gene2SubDifferCrossAttn(nn.Module):
    """
    Gene:        [B, P, G, E]
    Substructure:[B, S, E]
    Output:      [B, P, G, E]
    """
    def __init__(self, gene_embed_dim: int = 32, sub_embed_dim: int = 768, depth: int = 2):
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
        diff_attn_weights.sub_(min_value).add_(1e-5)

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
    def __init__(self, sub_embed_dim: int = 768, gene_embed_dim: int = 32, depth: int = 2):
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
        diff_attn_weights.sub_(min_value).add_(1e-5)

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

# Cross-Attention with Differential 2-Slot per Head
class Gene2SubDifferCrossMHA(nn.Module):
    """
    Gene:        [B, P, G, E]
    Substructure:[B, S, E]
    Output:      [B, P, G, E]
    """
    def __init__(self, gene_embed_dim: int, sub_embed_dim: int, attention_dim: int, num_heads: int, depth: int):
        super().__init__()
        # Conditions for MHA
        if attention_dim % (2 * num_heads) != 0:
            raise ValueError(
                f"embed_dim ({attention_dim}) must be divisible by 2 * num_heads ({2*num_heads})"
            )
        
        # Set dimensions
        self.gene_embed_dim = gene_embed_dim
        self.sub_embed_dim = sub_embed_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // (2 * num_heads)
        self.scaling = self.head_dim ** -0.5

        # Projection Layers
        self.q_proj = nn.Linear(gene_embed_dim,   2 * num_heads * self.head_dim, bias=False) # Query
        self.k_proj = nn.Linear(sub_embed_dim, 2 * num_heads * self.head_dim, bias=False) # Key
        self.v_proj = nn.Linear(sub_embed_dim, 2 * num_heads * self.head_dim, bias=False) # Value
        self.out_proj = nn.Linear(2 * num_heads * self.head_dim, gene_embed_dim, bias=False) # Output

         # Initialization λ
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        # RMS Norm after 2-slot head
        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)


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
        
        Lq = P * G
        query_flat = query.view(B, Lq, E)

        # Projection & Reshape to 2-slot head
        Q = (self.q_proj(query_flat)
             .view(B, Lq, self.num_heads, 2, self.head_dim)
             .permute(0, 2, 3, 1, 4))  # [B, H, 2, Lq, d]
        K = (self.k_proj(key)
             .view(B, S, self.num_heads, 2, self.head_dim)
             .permute(0, 2, 3, 1, 4))  # [B, H, 2, S, d]
        V = (self.v_proj(key)
             .view(B, S, self.num_heads, 2, self.head_dim)
             .permute(0, 2, 3, 1, 4))  # [B, H, 2, S, d]

        Q = Q * self.scaling

        scores = torch.matmul(Q, K.transpose(-1, -2))

        # Masking
        if query_mask is not None and key_mask is not None:
            query_mask = query_mask.to(Q.device)
            key_mask = key_mask.to(Q.device)
            query_mask_flat = query_mask.view(B, Lq)        # [B, Lq]
            combined_mask = query_mask_flat.unsqueeze(-1) & key_mask.unsqueeze(1)  # [B, Lq, S]
            combined_mask_exp = combined_mask.unsqueeze(1).unsqueeze(2)           # [B,1,1,Lq,S]
            combined_mask_exp = combined_mask_exp.expand(B, self.num_heads, 2, Lq, S)
            scores = scores.masked_fill(~combined_mask_exp, -1e20)

        # Compute Attention Scores
        scores = stable_softmax(scores, dim=-1)

        # Compute λ Scalar
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(Q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(Q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        
        # Differential Atention
        slot0 = scores[:, :, 0]
        slot1 = scores[:, :, 1]
        diff_attn = slot0 - lambda_full * slot1  # [B,H,Lq,S]

        # For Numerical Stability
        # min_value = diff_attn.min(dim=-1, keepdim=True).values 
        # diff_attn.sub_(min_value)

        # Concat V
        V_cat = (V.permute(0,1,3,2,4)
                 .reshape(B, self.num_heads, S, 2*self.head_dim))            # [B,H,S,2d]
        
        # Reshape for Output
        out = torch.matmul(diff_attn, V_cat)                               # [B,H,Lq,2d]
        out = self.subln(out) * (1 - self.lambda_init)  # RMS Norm
        out = (out.permute(0,2,1,3)
               .reshape(B, Lq, 2*self.num_heads*self.head_dim))
        out = self.out_proj(out).view(B, P, G, E)
        diff_attn = diff_attn.view(B, self.num_heads, P, G, S)

        return out, diff_attn
    


class Sub2GeneDifferCrossMHA(nn.Module):
    """
    Substructure: [B, S, E]
    Gene:         [B, P, G, E]
    Output:       [B, S, E]
    """
    def __init__(self, sub_embed_dim: int, gene_embed_dim: int, attention_dim: int, num_heads: int, depth: int):
        super().__init__()
        # Conditions for MHA
        if attention_dim % (2 * num_heads) != 0:
            raise ValueError(
                f"embed_dim ({attention_dim}) must be divisible by 2 * num_heads ({2*num_heads})"
            )

        # Set dimensions
        self.sub_embed_dim = sub_embed_dim
        self.gene_embed_dim = gene_embed_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // (2 * num_heads)
        self.scaling = self.head_dim ** -0.5

        # Projection Layers
        self.q_proj = nn.Linear(sub_embed_dim, 2 * num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(gene_embed_dim, 2 * num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(gene_embed_dim, 2 * num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(2 * num_heads * self.head_dim, sub_embed_dim, bias=False)

        # Initialization λ  
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))

        # RMS Norm after 2-slot head
        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, query: torch.Tensor, key: torch.Tensor, query_mask: torch.Tensor, key_mask: torch.Tensor):
        """
        query: [B, S, E]           # substructure
        key:   [B, P, G, E]        # gene
        """
        B, S, E = query.shape
        _, P, G, Ek = key.shape
        
        Lk = P * G
        key_flat = key.view(B, Lk, Ek)

        # Projection & Reshape to 2-slot head
        Q = self.q_proj(query).view(B, S, self.num_heads, 2, self.head_dim).permute(0, 2, 3, 1, 4)  # [B, H, 2, S, d]
        K = self.k_proj(key_flat).view(B, Lk, self.num_heads, 2, self.head_dim).permute(0, 2, 3, 1, 4)  # [B, H, 2, Lk, d]
        V = self.v_proj(key_flat).view(B, Lk, self.num_heads, 2, self.head_dim).permute(0, 2, 3, 1, 4)  # [B, H, 2, Lk, d]

        Q = Q * self.scaling

        scores = torch.matmul(Q, K.transpose(-1, -2))  # [B, H, 2, S, Lk]

        # Masking
        if query_mask is not None and key_mask is not None:
            query_mask = query_mask.to(Q.device)
            key_mask = key_mask.to(Q.device)
            key_mask_flat = key_mask.view(B, Lk)
            combined_mask = query_mask.unsqueeze(-1) & key_mask_flat.unsqueeze(1)  # [B, S, Lk]
            combined_mask_exp = combined_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S, Lk]
            combined_mask_exp = combined_mask_exp.expand(B, self.num_heads, 2, S, Lk)
            scores = scores.masked_fill(~combined_mask_exp, -1e20)

        # Compute Attention Scores
        scores = stable_softmax(scores, dim=-1)

        # Compute λ Scalar
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1)).type_as(Q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1)).type_as(Q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init  # scalar

        # Differential Atention
        slot0 = scores[:, :, 0]  # [B, H, S, Lk]
        slot1 = scores[:, :, 1]
        diff_attn = slot0 - lambda_full * slot1

        # Concat V
        V_cat = V.permute(0, 1, 3, 2, 4).reshape(B, self.num_heads, Lk, 2 * self.head_dim)  # [B, H, Lk, 2d]

        # Reshape for Output
        out = torch.matmul(diff_attn, V_cat)  # [B, H, S, 2d]
        out = self.subln(out) * (1 - self.lambda_init)
        out = out.permute(0, 2, 1, 3).reshape(B, S, 2 * self.num_heads * self.head_dim)
        out = self.out_proj(out)

        diff_attn = diff_attn.view(B, self.num_heads, S, P, G)
        return out, diff_attn
