import math
import torch
import torch.nn as nn
import pytorch_lightning as pl

from utils import Config


class GPTBlock(nn.Module):
    def __init__(self, d_embed, block_size, n_heads=8, dropout=0.2) -> None:
        super().__init__()
        self.mlp_dict = nn.ModuleDict(
            dict(
                hidden_1=nn.Linear(d_embed, 4 * d_embed),
                activation=nn.GELU(),
                hidden_2=nn.Linear(4 * d_embed, d_embed),
                mlp_dropout=nn.Dropout(dropout),
            )
        )

        self.ln_1 = nn.LayerNorm(d_embed)
        self.ln_2 = nn.LayerNorm(d_embed)
        self.attn_dropout = nn.Dropout(dropout)
        self.attention = nn.MultiheadAttention(d_embed, n_heads, batch_first=True)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

        self.attn = lambda x: self.attn_dropout(
            self.attention(x, x, x, attn_mask=self.mask, need_weights=False)[0]
        )

        self.mlp = lambda x: self.mlp_dict.mlp_dropout(
            self.mlp_dict.hidden_2(self.mlp_dict.activation(self.mlp_dict.hidden_1(x)))
        )

    def forward(self, x) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTTransformer(pl.LightningModule):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

        create_block = lambda: GPTBlock(
            config.d_embed, config.seq_len, config.n_heads, config.dropout
        )

        self.pos_embeddings = nn.Embedding(config.seq_len, config.d_embed)
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_embed)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.Sequential(*[create_block() for _ in range(config.n_layers)])
        self.lm_head = nn.Linear(config.d_embed, config.vocab_size)

        # initialize weights - scaling factor is taken from GPT-2 paper.
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("hidden_2.weight"):
                scaling = math.sqrt(2 * config.n_layers)
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / scaling)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x):
        pos = torch.arange(0, x.size(-1), dtype=torch.long)
        pos = pos.unsqueeze(0).to(x.device)
        pos_embeddings = self.pos_embeddings(pos)
        token_embeddings = self.token_embeddings(x)
        x = token_embeddings + pos_embeddings
        x = self.dropout(x)
        x = self.layers(x)
        return self.lm_head(x)

    def training_step(self, batch, _):
        x, y = batch[:, :-1], batch[:, 1:]
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat.transpose(1, 2), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch[:, :-1], batch[:, 1:]
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat.transpose(1, 2), y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
