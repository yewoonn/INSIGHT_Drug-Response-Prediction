import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from rms_norm import RMSNorm

#  GENE2SUB VANILLA CROSS-ATTENTION
class Gene2SubCrossAttn(nn.Module):
    def __init__(self, query_dim, key_dim):
        super(Gene2SubCrossAttn, self).__init__()
        self.query_layer = nn.Linear(query_dim, query_dim, bias=False)
        self.key_layer = nn.Linear(key_dim, query_dim, bias=False)
        self.value_layer = nn.Linear(key_dim, query_dim, bias=False)

    def forward(self, query, key, mask=None):
        B, P, G, E = query.shape   # e.g. [4, 5, 245, 32]
        _, S, _ = key.shape
        L = P * G

        # 1) Flatten query: [B, P, G, E] -> [B, (P*G), E]
        Q = query.view(B, L, E)  # => [B, (P*G), E]

        # 2) Apply query_layer, key_layer, value_layer
        Q = self.query_layer(Q)  # => [B, (P*G), E]
        K = self.key_layer(key)  # => [B, S, E]
        V = self.value_layer(key)  # => [B, S, E]

        # 3) Attention Score = Q @ K^T
        # => matmul([B, L, E], [B, E, S]) -> [B, L, S]
        attn_scores = torch.matmul(Q, K.transpose(-1, -2))  # => [B, L, S]

        # 4) Apply mask
        if mask is not None:
            mask = mask.to(query.device)
            mask = mask.view(B, -1)  # => [B, L]
            extended_mask = mask.unsqueeze(-1)  # => [B, L, 1]
            attn_scores = attn_scores.masked_fill(~extended_mask, float(0))

        # 5) Softmax over attention scores
        attn_weights = F.softmax(attn_scores, dim=-1)  # => [B, L, S]

        # 6) Compute attention output: attn_weights @ V
        attn_out = torch.matmul(attn_weights, V)  # => [B, L, E]

        # 7) Reshape output back to [B, P, G, E]
        attn_out = attn_out.view(B, P, G, E)
        attn_weights = attn_weights.view(B, P, G, S)

        return attn_out, attn_weights

#  SUB2GENE VANILLA CROSS-ATTENTION
class Sub2GeneCrossAttn(nn.Module):
    def __init__(self, query_dim, key_dim):
        super(Sub2GeneCrossAttn, self).__init__()
        self.query_layer = nn.Linear(query_dim, query_dim, bias=False)
        self.key_layer   = nn.Linear(key_dim,   query_dim, bias=False)
        self.value_layer = nn.Linear(key_dim,   query_dim, bias=False)

    def forward(self, query, key, mask=None):
        """
        Args:
            query: [B, S, query_dim]
                - 예: Substructure 임베딩 (Batch, Substructure, EmbeddingDim)
            key:   [B, P, G, key_dim]
                - 예: Gene 임베딩 (Batch, Pathway, Gene, EmbeddingDim)
            mask:  [B, S] (bool) (Optional)
                - Substructure 쪽에서 무효 토큰이 있을 경우 True/False로 처리
                - 있으면 query의 S 차원에 대응, 없으면 None

        Returns:
            attn_out:     [B, S, query_dim]
            attn_weights: [B, S, P, G]
        """
        # 1) Shape 파악
        B, S, Qdim = query.shape        # ex: [4, 11, 32]
        _, P, G, Kdim = key.shape       # ex: [4, 5, 245, 32]
        L = P * G  # Pathway x Gene = 전체 Key 길이

        # 2) Q, K, V projection
        Q = self.query_layer(query)     # [B, S, query_dim]
        K = self.key_layer(key)         # [B, P, G, query_dim]
        V = self.value_layer(key)       # [B, P, G, query_dim]

        # 3) Key/Value Flatten: (P, G) -> L
        #    => [B, P*G, query_dim]
        K = K.view(B, L, -1)  # ex: [4, 5*245, 32]
        V = V.view(B, L, -1)  # ex: [4, 5*245, 32]

        # 4) Attention Score = Q @ K^T
        # => matmul([B, S, E], [B, E, L]) => [B, S, L]
        attn_scores = torch.matmul(Q, K.transpose(-1, -2))  # => [B, S, L]

        # 5) 마스크가 있다면 (Substructure 쪽) -> [B, S] -> unsqueeze(-1) => [B, S, 1]
        #    => attn_scores.shape=[B, S, L], 마스크가 True=유효, False=무효인 경우:
        if mask is not None:
            mask = mask.to(query.device)
            # 보통은 무효 위치를 -∞로 만들기 위해 ~mask로 masked_fill
            extended_mask = mask.unsqueeze(-1)  # [B, S, 1]
            attn_scores = attn_scores.masked_fill(~extended_mask, float(0))

        # 6) Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # => [B, S, L]

        # 7) Output
        # => matmul([B, S, L], [B, L, E]) => [B, S, E]
        attn_out = torch.matmul(attn_weights, V)       # => [B, S, query_dim]

        # 8) attn_weights를 [B, S, P, G]로 reshape
        attn_weights = attn_weights.view(B, S, P, G)

        return attn_out, attn_weights  