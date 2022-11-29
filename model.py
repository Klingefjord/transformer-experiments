import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tokenizer import create_tokenizer

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
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size)) == 0
        )
        self.mlp = lambda x: self.mlp_dict.mlp_dropout(
            self.mlp_dict.hidden_2(self.mlp_dict.activation(self.mlp_dict.hidden_1(x)))
        )

    def attn(self, x):
        return self.attn_dropout(
            self.attention(
                x,
                x,
                x,
                attn_mask=self.mask[: x.shape[1], : x.shape[1]],
                need_weights=False,
            )[0]
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

        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_embed)
        self.pos_embeddings = nn.Embedding(config.seq_len, config.d_embed)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.Sequential(*[create_block() for _ in range(config.n_layers)])
        self.ln = nn.LayerNorm(config.d_embed)

        n_params = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6,))

        self.lm_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)

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
        # positional embeddings
        pos = torch.arange(0, x.shape[1], dtype=torch.long)
        pos = pos.unsqueeze(0).to(x.device)
        pos_embeddings = self.pos_embeddings(pos)
        # token embeddings
        token_embeddings = self.token_embeddings(x)
        # sum embeddings
        x = token_embeddings + pos_embeddings

        x = self.dropout(x)
        x = self.layers(x)
        x = self.ln(x)
        x = self.lm_head(x)
        return x

    def training_step(self, batch, idx):
        x, y = batch[:, :-1], batch[:, 1:]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat.permute(0, 2, 1), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, idx):
        x, y = batch[:, :-1], batch[:, 1:]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat.permute(0, 2, 1), y)
        self.log("val_loss", loss)
        if idx == 0 and self.config.should_sample:
            print(self.generate_samples(prompt="Hello, "))
        return loss

    def configure_optimizers(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        decay = set()
        no_decay = set()
        for name, param in self.named_parameters():
            if hasattr(param, "requires_grad") and not param.requires_grad:
                continue
            if "weight" in name and "norm" not in name and "bn" not in name:
                decay.add(name)
            else:
                no_decay.add(name)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
        )
        return optimizer

    @torch.no_grad()
    def generate(
        self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.seq_len
                else idx[:, -self.config.seq_len :]
            )
            # forward the model te get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.no_grad()
    def generate_samples(self, prompt="", num_samples=1, steps=20, do_sample=True):
        """Generate text using the model."""
        print("Generating dostoyevsky samples...")

        tokenizer = create_tokenizer(
            "./data/dostoyevsky.vocab", "./data/dostoyevsky.bpe"
        )

        if prompt == "":
            # to create unconditional samples...
            # manually create a tensor with only the special <|endoftext|> token
            # similar to what openai's code does here https://github.com/openai/gpt-2/blob/master/src/generate_unconditional_samples.py
            x = torch.tensor([[tokenizer.encoder["<|endoftext|>"]]], dtype=torch.long)
        else:
            x = torch.tensor([tokenizer.encode(prompt)]).to(self.device)

        # we'll process all desired num_samples in a batch, so expand out the batch dim
        x = x.expand(num_samples, -1)

        # forward the model `steps` times to get samples, in a batch
        y = self.generate(x, max_new_tokens=steps, do_sample=do_sample, top_k=40)

        out = ""

        for i in range(num_samples):
            out += tokenizer.decode(y[i].cpu().squeeze().tolist())
            out += "\n"
            out += "-" * 80
            out += "\n"

        del tokenizer

        return out
