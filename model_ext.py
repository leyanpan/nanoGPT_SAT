"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_lambda=None):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.use_rotary_emb = config.use_rotary_emb
        self.rotary_emb = getattr(config, 'rotary_emb', None)
    
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention.")
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                      .view(1, 1, config.block_size, config.block_size)
            )
    
        # New options
        self.relu_attention = getattr(config, 'relu_attention', False)
        self.log_scale_attention = getattr(config, 'log_scale_attention', False)
        self.layer_lambda = layer_lambda  # Expected shape: (n_head,) or None

    def forward(self, x, position_ids=None, attn_mask=None, layer_lambda=None):
        """
        Forward pass for CausalSelfAttention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).
            position_ids (torch.Tensor, optional): Position indices of shape (B, T). Defaults to None.
            attn_mask (torch.Tensor, optional): Attention mask of shape (B, T), with 1 indicating positions to attend to and 0 otherwise. Defaults to None.
            layer_lambda (torch.Tensor, optional): Tensor of shape (n_head,) for log_scale_attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C).
        """
        B, T, C = x.size()

        # 1. Project x into Q, K, V
        qkv = self.c_attn(x)  # Shape: (B, T, 3 * C)
        q, k, v = qkv.split(self.n_embd, dim=2)  # Each of shape: (B, T, C)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # Shape: (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # Shape: (B, n_head, T, head_dim)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # Shape: (B, n_head, T, head_dim)

        # 2. Apply rotary embeddings if requested
        if self.use_rotary_emb:
            if position_ids is None:
                position_ids = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0).expand(B, T)
            cos, sin = self.rotary_emb(q, position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 3. Compute attention scores
        att_scores = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))  # Shape: (B, n_head, T, T)

        # 4. Apply log_scale_attention if enabled
        if self.log_scale_attention and layer_lambda is not None:
            # layer_lambda: shape (n_head,)
            # Compute log(i + 1) for positions 0..T-1 to avoid log(0)
            pos_ids = torch.arange(0, T, device=x.device).float() + 1.0  # Shape: (T,)
            log_pos = torch.log(pos_ids)  # Shape: (T,)
            # Expand dimensions to match att_scores: (1, n_head, T, 1)
            log_pos = log_pos.view(1, 1, T, 1)
            # Expand layer_lambda to (1, n_head, 1, 1)
            lamb = layer_lambda.view(1, self.n_head, 1, 1)
            # Scale attention scores
            att_scores = att_scores * (1.0 + lamb * log_pos)  # Shape: (B, n_head, T, T)

        # 5. Create causal mask (lower-triangular)
        causal_mask = torch.tril(torch.ones((T, T), device=x.device)).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, T, T)

        # 6. Combine causal mask with attn_mask
        if attn_mask is not None:
            # attn_mask: (B, T) with 1 for attend and 0 for not attend
            # Expand attn_mask to (B, 1, 1, T)
            attn_mask_expanded = attn_mask.view(B, 1, 1, T)
            # Combine masks: only attend to positions where both causal_mask and attn_mask are 1
            combined_mask = causal_mask * attn_mask_expanded  # Shape: (B, 1, T, T)
        else:
            combined_mask = causal_mask  # Shape: (1, 1, T, T)

        # 7. Handle relu_attention
        if self.relu_attention:
            # Apply ReLU activation
            att_scores = F.relu(att_scores)  # Shape: (B, n_head, T, T)
            # Apply combined mask: set positions not to attend to as 0
            att_scores = att_scores * combined_mask  # Zero out masked positions
            # Normalize attention weights so that they sum to 1 along the last dimension
            att_weights = att_scores / (att_scores.sum(dim=-1, keepdim=True) + 1e-9)  # Shape: (B, n_head, T, T)
        else:
            # For standard softmax-based attention
            # Apply mask by adding -1e9 to masked positions
            # combined_mask: 1 for attend, 0 for not attend
            att_scores = att_scores.masked_fill(combined_mask == 0, -1e9)  # Shape: (B, n_head, T, T)
            # Apply softmax
            att_weights = F.softmax(att_scores, dim=-1)  # Shape: (B, n_head, T, T)

        # 8. Apply dropout to attention weights
        att_weights = self.attn_dropout(att_weights)  # Shape: (B, n_head, T, T)

        # 9. Compute attention output
        att_output = att_weights @ v  # Shape: (B, n_head, T, head_dim)

        # 10. Merge heads and project
        att_output = att_output.transpose(1, 2).contiguous().view(B, T, C)  # Shape: (B, T, C)
        att_output = self.c_proj(att_output)  # Shape: (B, T, C)

        return att_output


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # We do an 8× expansion, then split into two 4× halves (SwiGLU),
        # then project from 4× back to 1×.
        self.c_fc = nn.Linear(config.n_embd, 8 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        # Step 1: expand from n_embd -> 8*n_embd
        x_fc = self.c_fc(x)  # shape: (batch, seq_len, 8*n_embd)

        # Step 2: split into two 4× parts (a, b)
        a, b = x_fc.split(x_fc.size(-1) // 2, dim=-1)  # each (4*n_embd)

        # Step 3: SwiGLU
        x = F.silu(b)  # shape: (batch, seq_len, 4*n_embd)

        # Step 4: project back to n_embd
        x = self.c_proj(x)     # shape: (batch, seq_len, n_embd)

        return x

class Block(nn.Module):
    def __init__(self, config, layer_idx=None, lambda_params=None):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.n_embd, eps=1e-6, elementwise_affine=True)
        
        # Pass layer_lambda to CausalSelfAttention if provided
        self.attn = CausalSelfAttention(config, layer_lambda=lambda_params)
    
        self.post_attention_layernorm = nn.RMSNorm(config.n_embd, eps=1e-6, elementwise_affine=True)
        self.mlp = MLP(config)

    def forward(self, x, position_ids=None, attn_mask=None):
        h = self.input_layernorm(x)
        x = x + self.attn(h, position_ids=position_ids, attn_mask=attn_mask)

        h2 = self.post_attention_layernorm(x)
        x = x + self.mlp(h2)
        return x

class LSTMBlock(nn.Module):
    """
    A block that uses LSTM instead of self-attention, but keeps the same MLP afterward.
    LN -> LSTM -> residual, LN -> MLP -> residual
    """

    def __init__(self, config):
        super().__init__()
        # Instead of ln_1 + CausalSelfAttention, we have ln_1 + LSTM
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.lstm = nn.LSTM(
            input_size=config.n_embd,
            hidden_size=config.n_embd,
            batch_first=True,   # (B, T, C)
        )

        # The second LN + MLP is the same as in the original Block
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, position_ids=None, attn_mask=None):
        # 1) LN -> LSTM -> Residual
        h = self.ln_1(x)  # (B, T, C)
        out, _ = self.lstm(h)  # out.shape = (B, T, C) by default
        x = x + out

        # 2) LN -> MLP -> Residual
        h2 = self.ln_2(x)
        x = x + self.mlp(h2)
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    num_lstm_layers: int = 0
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    use_rotary_emb: bool = False  # NEW: Use rotary embeddings in attention layers
    relu_attention: bool = False  # NEW: Use ReLU activation in attention scores
    log_scale_attention: bool = False

    def to_dict(self):
        return asdict(self)

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # === Embeddings and transformer container ===
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd) if config.num_lstm_layers == 0 and not config.use_rotary_emb else None,
            drop = nn.Dropout(config.dropout),
        ))

        # Initialize lambda parameters for log_scale_attention
        if getattr(config, 'log_scale_attention', False):
            # Initialize lambda_params as a learnable parameter of shape (n_layer, n_head)
            self.lambda_params = nn.Parameter(torch.ones(config.n_layer, config.n_head))

        # Build the stack of blocks (mix of LSTMBlock + Block)
        blocks = []
        for i in range(config.n_layer):
            if i < config.num_lstm_layers:
                # Use LSTM-based block
                blocks.append(LSTMBlock(config))
            else:
                # Use standard Transformer block, passing layer index and lambda_params if needed
                blocks.append(Block(
                    config,
                    layer_idx=i,
                    lambda_params=self.lambda_params[i] if getattr(config, 'log_scale_attention', False) else None
                ))

        self.transformer["h"] = nn.ModuleList(blocks)

        # Final layernorm
        self.transformer["ln_f"] = LayerNorm(config.n_embd, bias=config.bias)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # If using rotary embeddings, attach them
        if config.use_rotary_emb:
            head_dim = config.n_embd // config.n_head
            self.rotary_emb = LlamaRotaryEmbedding(
                dim=head_dim,
                max_position_embeddings=config.block_size,
                base=10000,
                scaling_factor=1.0,
                rope_type="default",
                config=None
            )
            # Assign to each Transformer block that uses self-attn
            # (LSTMBlocks do not require it, so skip them)
            for block in self.transformer.h:
                if isinstance(block, Block):
                    block.attn.rotary_emb = self.rotary_emb

        # Weight initialization
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))


    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.transformer.wpe:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, attention_mask=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # If not using rotary, we add position embeddings
        if self.transformer.wpe is not None:
            pos = torch.arange(0, t, dtype=torch.long, device=device)  # (t,)
            tok_emb = self.transformer.wte(idx)        # (B, T, C)
            pos_emb = self.transformer.wpe(pos)        # (T, C)
            x = tok_emb + pos_emb.unsqueeze(0)         # broadcast along batch
            position_ids = None
        else:
            x = self.transformer.wte(idx)
            position_ids = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0).expand(b, t)


        # Pass through the stack of blocks
        for block in self.transformer.h:
            x = block(x, position_ids=position_ids, attn_mask=attention_mask)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            # SHIFT logic
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_targets = targets[:, 1:].contiguous()
            loss = F.cross_entropy(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_targets.view(-1),
                ignore_index=-100
            )
        else:
            logits = logits[:, [-1], :]  # only return last position
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None, stop=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = input_ids if input_ids.size(1) <= self.config.block_size else input_ids[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)

            if temperature == 0:
                # Greedy decoding: directly choose the token with the highest probability
                # no temperature scaling or sampling
                logits = logits[:, -1, :]
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # Temperature-based decoding
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)

            # append the chosen token index to the running sequence
            input_ids = torch.cat((input_ids, idx_next), dim=1)

            # if we sampled the stop token, return the sequence
            if stop is not None and idx_next.item() in stop:
                break

        return input_ids


    
    @torch.no_grad()
    def classify(self, idx):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and use the first position to predict a single token.
        """
        logits, _ = self(idx)
        logits = logits[:, 0, :]
        return logits.argmax(dim=-1)

    @property
    def device(self):
        return next(self.parameters()).device
    
    def summary(self) -> str:
        """
        Returns a string summary of the core model configuration and
        parameter statistics, which you can compare to a Hugging Face LLaMA model.
        """
        # Extract relevant fields from self.config
        n_layer = getattr(self.config, 'n_layer', 'N/A')
        n_head = getattr(self.config, 'n_head', 'N/A')
        n_embd = getattr(self.config, 'n_embd', 'N/A')
        vocab_size = getattr(self.config, 'vocab_size', 'N/A')
        block_size = getattr(self.config, 'block_size', 'N/A')
        dropout = getattr(self.config, 'dropout', 'N/A')
        bias = getattr(self.config, 'bias', 'N/A')

        # Compute total parameters
        total_params = self.get_num_params(non_embedding=False)
        total_params_nonembed = self.get_num_params(non_embedding=True)

        # Build a summary string
        summary_str = (
            "======== Model Summary ========\n"
            f"Model class: {self.__class__.__name__}\n"
            f"Number of layers (n_layer): {n_layer}\n"
            f"Hidden size (n_embd): {n_embd}\n"
            f"Number of attention heads (n_head): {n_head}\n"
            f"Vocab size (vocab_size): {vocab_size}\n"
            f"Block size / context window (block_size): {block_size}\n"
            f"Dropout: {dropout}\n"
            f"Bias in linear/LN layers: {bias}\n"
            f"Total parameters (including embeddings): {total_params:,}\n"
            f"Total parameters (excluding positional embeddings): {total_params_nonembed:,}\n"
            "===============================\n"
        )
        return summary_str

