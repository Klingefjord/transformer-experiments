import torch
import torch.nn as nn


class GPTBlock(nn.Module):
    def __init__(self, d_embed, block_size, n_heads=8, dropout=0.2) -> None:
        super().__init__()
        mlp = [
            nn.Linear(d_embed, 4 * d_embed),
            nn.GELU(),
            nn.Linear(4 * d_embed, d_embed),
            nn.Dropout(dropout),
        ]
        self.ln_1 = nn.LayerNorm(d_embed)
        self.attention = nn.MultiheadAttention(d_embed, n_heads, batch_first=True)
        self.ln_2 = nn.LayerNorm(d_embed)
        self.mlp = nn.Sequential(*mlp)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )

        self.attn = lambda x: self.dropout(self.attention(x, attn_mask=self.mask))

    def forward(self, x) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_embed=256,
        block_size=1024,
        n_layers=6,
        n_heads=8,
        dropout=0.2,
    ) -> None:
        super().__init__()
        self.pos_embeddings = nn.Embedding(block_size, d_embed)
        self.token_embeddings = nn.Embedding(vocab_size, d_embed)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.Sequential(
            [GPTBlock(d_embed, block_size, n_heads, dropout) for _ in range(n_layers)]
        )
        self.lm_head = nn.Linear(d_embed, vocab_size)

    def forward(self, x):
        x = self.token_embeddings(x)
        x = self.pos_embeddings(x)
        x = self.layers(x)
        return self.lm_head(x)
