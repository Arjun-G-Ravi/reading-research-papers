import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    """
    Simple CNN encoder that converts a 32x32 image into a grid of tokens (flattened spatial features).
    Output shape: (B, N, D), where N = H*W after downsampling.
    """
    def __init__(self, in_ch=3, width=64, depth=3, out_dim=128):
        super().__init__()
        layers = []
        ch = in_ch
        for i in range(depth):
            layers += [
                nn.Conv2d(ch, width, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.Conv2d(width, width, kernel_size=3, stride=2 if i < depth - 1 else 1, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
            ]
            ch = width
            width = max(width, width)  # keep width
        self.net = nn.Sequential(*layers)
        self.proj = nn.Conv2d(ch, out_dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        # x: (B, 3, 32, 32)
        feats = self.net(x)         # (B, C, H', W')
        feats = self.proj(feats)    # (B, D, H', W')
        B, D, H, W = feats.shape
        tokens = feats.permute(0, 2, 3, 1).reshape(B, H * W, D)  # (B, N, D)
        return tokens, (H, W)


class TokenAttention(nn.Module):
    """
    Single-head scaled dot-product attention over tokens.
    Query q from controller state, keys/values from tokens.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, q: torch.Tensor, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q: (B, D), tokens: (B, N, D)
        B, N, D = tokens.shape
        q_proj = self.q_proj(q).unsqueeze(1)           # (B, 1, D)
        k_proj = self.k_proj(tokens)                   # (B, N, D)
        v_proj = self.v_proj(tokens)                   # (B, N, D)
        attn_scores = torch.bmm(q_proj, k_proj.transpose(1, 2)).squeeze(1)  # (B, N)
        attn_scores = attn_scores / math.sqrt(D)
        attn = F.softmax(attn_scores, dim=-1)          # (B, N)
        context = torch.bmm(attn.unsqueeze(1), v_proj).squeeze(1)  # (B, D)
        return context, attn


class HighLevelController(nn.Module):
    """
    High-level recurrent controller (macro-steps) that produces queries/plans.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRUCell(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim), h: (B, hidden_dim)
        return self.gru(x, h)


class LowLevelExecutor(nn.Module):
    """
    Low-level executor (micro-steps) that refines memory given a context vector.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, context: torch.Tensor, s: torch.Tensor, steps: int = 2) -> torch.Tensor:
        # context: (B, D), s: (B, H)
        for _ in range(steps):
            s_new = self.gru(context, s)
            g = self.gate(torch.cat([s_new, context], dim=-1))
            s = s + g * (s_new - s)  # gated residual update
        return s


class HRMClassifier(nn.Module):
    """
    Minimal HRM-style classifier:
      - Encode image to tokens
      - For T macro steps:
          - High-level controller updates its state
          - Attend over tokens using the controller state
          - Low-level executor refines a memory state using the attended context (K micro steps)
      - Classify from final memory
    """
    def __init__(
        self,
        num_classes: int = 10,
        token_dim: int = 128,
        ctrl_dim: int = 256,
        mem_dim: int = 256,
        macro_steps: int = 3,
        micro_steps: int = 2,
        cnn_width: int = 64,
        cnn_depth: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = ImageEncoder(in_ch=3, width=cnn_width, depth=cnn_depth, out_dim=token_dim)
        self.controller = HighLevelController(input_dim=token_dim + mem_dim, hidden_dim=ctrl_dim)
        self.attn = TokenAttention(dim=token_dim)
        self.executor = LowLevelExecutor(input_dim=token_dim, hidden_dim=mem_dim)
        self.macro_steps = macro_steps
        self.micro_steps = micro_steps

        self.mem_init = nn.Parameter(torch.zeros(1, mem_dim))
        self.ctrl_init = nn.Parameter(torch.zeros(1, ctrl_dim))

        self.fuse = nn.Sequential(
            nn.Linear(ctrl_dim + mem_dim, mem_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(mem_dim, num_classes)

        # Project controller state to token space for attention queries if dimensions differ
        self.ctrl_to_token = nn.Linear(ctrl_dim, token_dim, bias=False)

        # Summarize tokens for controller input
        self.token_pool = nn.Linear(token_dim, token_dim)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B = x.size(0)
        tokens, _ = self.encoder(x)             # (B, N, D)
        token_summary = self.token_pool(tokens.mean(dim=1))  # (B, D)

        mem = self.mem_init.expand(B, -1).contiguous()   # (B, M)
        ctrl = self.ctrl_init.expand(B, -1).contiguous() # (B, C)

        all_attn = []

        for _ in range(self.macro_steps):
            ctrl_inp = torch.cat([token_summary, mem], dim=-1)  # (B, D+M)
            ctrl = self.controller(ctrl_inp, ctrl)              # (B, C)

            q = self.ctrl_to_token(ctrl)                       # (B, D)
            context, attn = self.attn(q, tokens)               # (B, D), (B, N)
            if return_attn:
                all_attn.append(attn)

            mem_new = self.executor(context, mem, steps=self.micro_steps)  # (B, M)
            mem = self.fuse(torch.cat([ctrl, mem_new], dim=-1))            # (B, M)

        logits = self.head(mem)  # (B, num_classes)
        if return_attn:
            return logits, torch.stack(all_attn, dim=1)  # (B, T, N)
        return logits