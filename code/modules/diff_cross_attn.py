import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from modules.rms_norm import RMSNorm
 
#  DIFFERENTIAL CROSS-ATTENTION
class DifferCrossAttn(nn.Module):
    def __init__(self, embed_dim: int, depth: int = 0):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // 2
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)  # (E -> E)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)  # (E -> E)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)  # (E -> E)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)

        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, gene: torch.Tensor, substructure: torch.Tensor) -> torch.Tensor:
        bsz, n_gene, _ = gene.size()
        _, n_sub,  _ = substructure.size()

        # 1) Q, K, V 구하기 (Cross Attention)
        q = self.q_proj(gene)              # (B, n_gene, E)
        k = self.k_proj(substructure)      # (B, n_sub,  E)
        v = self.v_proj(substructure)      # (B, n_sub,  E)

        # 2) Q, K -> [2, head_dim], V -> [2 * head_dim]
        q = q.view(bsz, n_gene, 2, self.head_dim)   # (B, n_gene, 2, head_dim)
        k = k.view(bsz, n_sub,  2, self.head_dim)   # (B, n_sub,  2, head_dim)
        v = v.view(bsz, n_sub,  2 * self.head_dim)  # (B, n_sub,  2 * head_dim)

        # 3) matmul 위해 transpose (slot 차원을 두 번째로)
        q = q.transpose(1, 2)  # (B, 2, n_gene, head_dim)
        k = k.transpose(1, 2)  # (B, 2, n_sub,  head_dim)
        v = v.unsqueeze(1)     # (B, 1, n_sub, 2*head_dim)

        q = q * self.scaling

        # 4) attention score = Q @ K^T
        #    => (B, 2, n_gene, head_dim) x (B, 2, head_dim, n_sub) = (B, 2, n_gene, n_sub)
        attn_weights = torch.matmul(q, k.transpose(-1, -2))  # (B, 2, n_gene, n_sub)
        attn_weights = F.softmax(attn_weights, dim=-1)  # (B, 2, n_gene, n_sub)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1))
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # 5) 두 슬롯(0, 1) 간 차이를 내어 최종 attn_weights 계산
        diff_attn_weights = attn_weights[:, 0] - lambda_full * attn_weights[:, 1]  # (B, n_gene, n_sub)
        diff_attn = diff_attn_weights.unsqueeze(1)  # (B, 1, n_gene, n_sub)

        attn_output = torch.matmul(diff_attn, v)
        attn_output = self.subln(attn_output)               # (B, 1, n_gene, 2*head_dim)
        attn_output = attn_output * (1.0 - self.lambda_init)
        attn_output = attn_output.squeeze(1)                # (B, n_gene, 2*head_dim)
        out = self.out_proj(attn_output)                    # (B, n_gene, E)

        return out, diff_attn_weights

#  GENE2SUB DIFFERENTIAL CROSS-ATTENTION
class Gene2SubDifferCrossAttn(nn.Module):
    """
    Gene:        [B, P, G, E]
    Substructure:[B, S, E]
    Output:      [B, P, G, E]
    """
    def __init__(self, embed_dim: int, depth: int = 0):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // 2  
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, query: torch.Tensor, key: torch.Tensor, mask):
        """
        gene:        [B, P, G, E]
        substructure:[B, S, E]
        Returns:
            out:               [B, P, G, E]
            diff_attn_weights: [B, P, G, S]
        """
        B, P, G, E = query.shape  # [4, 5, 245, 32]
        S = key.size(1)

        # 1) Flatten gene -> [B, L, E],  L = P*G
        L = P * G
        gene_flat = query.view(B, L, E)  # [B, (P*G), E]

        # 2) Q, K, V
        Q = self.q_proj(gene_flat)      # [B, L, E]
        K = self.k_proj(key)   # [B, S, E]
        V = self.v_proj(key)   # [B, S, E]

        # 3) 2-Slot(Head)으로 분리
        #    Q => [B, L, 2, head_dim]
        #    K => [B, S, 2, head_dim]
        #    V => [B, S, 2 * head_dim]
        Q = Q.view(B, L, 2, self.head_dim).transpose(1, 2)  # => [B, 2, L, head_dim]
        K = K.view(B, S, 2, self.head_dim).transpose(1, 2)  # => [B, 2, S, head_dim]
        V = V.view(B, S, 2 * self.head_dim)                 # => [B, S, 2*head_dim]

        # 4) Scaling & Attention Score
        Q = Q * self.scaling
        # => [B, 2, L, head_dim] x [B, 2, head_dim, S] => [B, 2, L, S]
        attn_scores = torch.matmul(Q, K.transpose(-1, -2))

        if mask is not None:
            # mask: [B, P, G] => Flatten -> [B, L]
            mask = mask.to(query.device)
            mask_flat = mask.view(B, L)  # [B, L]
            # attn_scores: [B, 2, L, S]
            # 확장을 위해 mask_flat => [B, 1, L, 1]
            extended_mask = mask_flat.unsqueeze(1).unsqueeze(-1)  # [B, 1, L, 1]
            # 무효 위치(False) = -1e9
            attn_scores = attn_scores.masked_fill(~extended_mask, float(0))

        attn_scores = F.softmax(attn_scores, dim=-1)  # [B, 2, L, S]

        # 5) 두 슬롯(0,1) 차이 계산
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1))
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # [B, 2, L, S]
        diff_attn_weights = attn_scores[:, 0] - lambda_full * attn_scores[:, 1]  # => [B, L, S]

        # 6) V와 곱
        # diff_attn_weights: [B, L, S] -> [B, 1, L, S]
        # 여기서는 bmm 형태로 쉽게 하기 위해 2D 변환
        # => matmul([B, L, S], [B, S, 2*head_dim]) => [B, L, 2*head_dim]
        diff_attn_2d = diff_attn_weights             # [B, L, S]
        attn_out_2d = torch.bmm(diff_attn_2d, V)     # [B, L, 2*head_dim]

        attn_out_2d = attn_out_2d.unsqueeze(1)       # => [B, 1, L, 2*head_dim]
        attn_out_2d = self.subln(attn_out_2d)        # => [B, 1, L, 2*head_dim]
        attn_out_2d = attn_out_2d * (1.0 - self.lambda_init)

        # 7) 최종 Linear
        attn_out_2d = attn_out_2d.squeeze(1)         # => [B, L, 2*head_dim]
        out = self.out_proj(attn_out_2d)            # => [B, L, E]

        # 8) 원본 shape로 복원
        out = out.view(B, P, G, E)                  # => [B, P, G, E]

        # diff_attn_weights => [B, L, S] -> [B, P, G, S]
        diff_attn_weights = diff_attn_weights.view(B, P, G, S)

        return out, diff_attn_weights

#  SUB2GENE DIFFERENTIAL CROSS-ATTENTION
class Sub2GeneDifferCrossAttn(nn.Module):
    """
    Substructure: [B, S, E]
    Gene:         [B, P, G, E]
    Output:       [B, S, E]
    """
    def __init__(self, embed_dim: int, depth: int = 0):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // 2
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, query: torch.Tensor, key: torch.Tensor, mask):
        """
        substructure: [B, S, E]
        gene:         [B, P, G, E]
        Returns:
            out:               [B, S, E]
            diff_attn_weights: [B, S, P, G]
        """
        B, S, E = query.shape
        _, P, G, _ = key.shape
        L = P * G  # Flatten 길이

        # 1) Q, K, V 투영
        #    - Query: substructure, Key/Value: gene
        Q = self.q_proj(query)  # [B, S, E]
        K = self.k_proj(key)         # [B, P, G, E]
        V = self.v_proj(key)         # [B, P, G, E]

        # 2) Flatten gene => [B, L, E],  L = P*G
        K = K.view(B, L, E)
        V = V.view(B, L, 2 * (self.head_dim)) if (2 * self.head_dim == E) else V.view(B, L, E)
        # 주의: 여기에서 V도 "2 * head_dim" == E 인지 여부에 따라 달라질 수 있습니다.
        # 만약 embed_dim과 Flatten 로직이 정확히 매칭되면 괜찮지만,
        # slot 2개로 쪼개기 위해서 K, V도 분리해야 한다면 아래처럼 해야 합니다.
        # => 만약 완전히 동일하게 '2개 슬롯'을 gene에 적용하려면,
        #    gene 자체가 [B, P, G, E]에서 E가 "2*head_dim"이어야 맞습니다.
        # 일단 여기서는 E 그대로 projection했다고 가정하면
        #   K => [B, L, E]
        #   V => [B, L, E]
        # 라고 보고, 아래에서 다시 2, head_dim으로 분리하겠습니다.

        # 3) 2-Slot(Head) 분리
        #    Q => [B, S, E] -> [B, S, 2, head_dim]
        Q = Q.view(B, S, 2, self.head_dim).transpose(1, 2)  # => [B, 2, S, head_dim]
        K = K.view(B, L, 2, self.head_dim).transpose(1, 2)  # => [B, 2, L, head_dim]
        V = V.view(B, L, 2 * self.head_dim)                 # => [B, L, 2*head_dim]

        # 4) Scaling + Attention Score
        Q = Q * self.scaling
        # => [B, 2, S, head_dim] x [B, 2, head_dim, L] => [B, 2, S, L]
        attn_scores = torch.matmul(Q, K.transpose(-1, -2))

        if mask is not None:
            # mask: [B, S]
            # attn_scores: [B, 2, S, L]
            # 확장 => [B, 1, S, 1]
            mask = mask.to(query.device)
            extended_mask = mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, S, 1]
            attn_scores = attn_scores.masked_fill(~extended_mask, float(0))


        attn_scores = F.softmax(attn_scores, dim=-1)  # [B, 2, S, L]

        # 5) λ 계산 후 차이
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1))
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        diff_attn_weights = attn_scores[:, 0] - lambda_full * attn_scores[:, 1]  # => [B, S, L]

        # 6) diff_attn_weights * V
        attn_out_2d = torch.bmm(diff_attn_weights, V)  # [B, S, 2*head_dim]

        attn_out_2d = attn_out_2d.unsqueeze(1)         # [B, 1, S, 2*head_dim]
        attn_out_2d = self.subln(attn_out_2d)          # [B, 1, S, 2*head_dim]
        attn_out_2d = attn_out_2d * (1.0 - self.lambda_init)

        # 7) 최종 out_proj
        attn_out_2d = attn_out_2d.squeeze(1)           # [B, S, 2*head_dim]
        out = self.out_proj(attn_out_2d)               # [B, S, E]

        # 8) diff_attn_weights -> [B, S, P, G]
        diff_attn_weights = diff_attn_weights.view(B, S, P, G)

        return out, diff_attn_weights