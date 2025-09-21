from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

# Custom modules for initialization and layers
from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

# === DATA STRUCTURES ===

@dataclass
class HierarchicalReasoningModel_ACTV1InnerCarry:
    """
    Stores the inner carry state for the hierarchical reasoning model.
    z_H: State at the high (H) reasoning level.
    z_L: State at the low (L) reasoning level.
    """
    z_H: torch.Tensor
    z_L: torch.Tensor

@dataclass
class HierarchicalReasoningModel_ACTV1Carry:
    """
    Stores the carry state for the ACT (Adaptive Computation Time) reasoning model.
    inner_carry: Internal carry state (high/low levels).
    steps: Number of reasoning steps taken for each item in the batch.
    halted: Boolean mask for which batch elements have halted computation.
    current_data: Dictionary holding current batch data.
    """
    inner_carry: HierarchicalReasoningModel_ACTV1InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class HierarchicalReasoningModel_ACTV1Config(BaseModel):
    """
    Configuration for the hierarchical reasoning model.
    Defines model, transformer, and halting hyperparameters.
    """
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

# === MODEL BLOCKS ===

class HierarchicalReasoningModel_ACTV1Block(nn.Module):
    """
    One transformer-like reasoning block: Attention + SwiGLU + RMSNorm.
    Used for both H (high) and L (low) levels.
    """
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False  # Not causal: can attend bidirectionally
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post-norm transformer block: add residual, apply attention, then MLP + normalization
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps
        )
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states),
            variance_epsilon=self.norm_eps
        )
        return hidden_states

class HierarchicalReasoningModel_ACTV1ReasoningModule(nn.Module):
    """
    Sequence of reasoning blocks (either H or L level).
    Supports input injection (additive).
    """
    def __init__(self, layers: List[HierarchicalReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # Inject additional input at the start (e.g., cross-level context)
        hidden_states = hidden_states + input_injection
        # Apply stacked transformer-like blocks
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states

# === ACTV1 INNER MODEL ===

class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):
    """
    The core hierarchical model:
    - Handles token & puzzle embeddings, positional encodings, transformer blocks
    - Supports two-level reasoning: H (high) and L (low)
    - Q-head for halting via reinforcement learning
    """
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # --- Embedding layers ---
        self.embed_scale  = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        # Token embedding
        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype
        )
        # Output head (language modeling)
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # Q-head (binary: halt/continue)
        self.q_head  = CastedLinear(self.config.hidden_size, 2, bias=True)

        # Puzzle embeddings (optional, for extra context/conditioning)
        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype
            )

        # --- Positional encoding ---
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype
            )
        else:
            raise NotImplementedError('Unknown positional encoding type')

        # --- Reasoning modules (stacks of blocks) ---
        self.H_level = HierarchicalReasoningModel_ACTV1ReasoningModule(
            layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.H_layers)]
        )
        self.L_level = HierarchicalReasoningModel_ACTV1ReasoningModule(
            layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)]
        )
        
        # --- Initial states for H and L ---
        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True
        )

        # --- Q-head initialization (for RL training stability) ---
        with torch.no_grad():
            self.q_head.weight.zero_()                  # Start Q logits at 0
            self.q_head.bias.fill_(-5)                  # Very negative bias, so model learns to halt slowly

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        """
        Compose input embedding:
        - Token embedding
        - (Optional) Puzzle embedding, zero-padded & concatenated
        - Add position embedding (if learned)
        - Scale
        """
        embedding = self.embed_tokens(input.to(torch.int32))

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            # Pad the puzzle embedding to fit hidden size
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            # Concatenate puzzle embedding at start of sequence
            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding),
                dim=-2
            )

        if self.config.pos_encodings == "learned":
            # Add learned position embedding, scaled for variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        """
        Returns a blank carry state for a batch (no gradients).
        """
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: HierarchicalReasoningModel_ACTV1InnerCarry):
        """
        Resets the carry state for sequences where reset_flag is true.
        """
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(
        self,
        carry: HierarchicalReasoningModel_ACTV1InnerCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[HierarchicalReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for the hierarchical reasoning model (1 outer step).
        - Applies L-cycles within each H-cycle, using input injection.
        - Detaches state for 1-step gradient.
        - Computes logits and Q-values for halting.
        Returns new carry state, model logits, and Q logits.
        """
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Get embeddings for input tokens and puzzle identifiers
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # --- Hierarchical reasoning (no grad pass for all but last step) ---
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L
            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    # Run all but the last (H, L) step with no gradient
                    if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                if not (_H_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, **seq_info)
        assert not z_H.requires_grad and not z_L.requires_grad

        # --- 1-step gradient for final (H, L) pass ---
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        # --- Output heads ---
        # Detach carry state for next iteration
        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        # Language modeling logits (skip puzzle embedding tokens)
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        # Q-head logits for halting (use first token of z_H)
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])

# === ACT WRAPPER ===

class HierarchicalReasoningModel_ACTV1(nn.Module):
    """
    Adaptive Computation Time (ACT) wrapper for the hierarchical model.
    Manages halting, steps, and batch carry logic.
    """
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV1Config(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        # Expose puzzle embedding for outside use
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        """
        Initializes the outer carry state for the batch.
        All sequences start as halted (to be reset in first pass).
        """
        batch_size = batch["inputs"].shape[0]
        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # All halted initially
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(
        self,
        carry: HierarchicalReasoningModel_ACTV1Carry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        """
        Outer ACT forward step.
        - Resets inner carry for halted sequences
        - Updates step counts, halting masks, and current batch data
        - Computes outputs and Q-values for halting
        - Optionally computes target Q for RL training
        Returns new carry and output dictionary.
        """
        # Update carry state for halted sequences
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        # Reset steps for halted, keep steps for non-halted
        new_steps = torch.where(carry.halted, 0, carry.steps)
        # Update batch data for halted sequences
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)),
                batch[k],
                v
            )
            for k, v in carry.current_data.items()
        }

        # Forward pass of the hierarchical model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        
        with torch.no_grad():
            # Increment step count
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step

            # If training and ACT enabled: use Q-head to determine halting
            if self.training and (self.config.halt_max_steps > 1):
                # Halt if Q_halt > Q_continue, or if max steps reached
                halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration: force minimum halting steps with some probability
                min_halt_steps = (
                    (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob)
                    * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                )
                halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q value (for RL loss): next Q logits
                # Use maximum of next Q_halt/Q_continue unless at last step
                next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry, new_current_data)[-1]
                outputs["target_q_continue"] = torch.sigmoid(
                    torch.where(
                        is_last_step,
                        next_q_halt_logits,
                        torch.maximum(next_q_halt_logits, next_q_continue_logits)
                    )
                )

        # Return updated carry and outputs
        return HierarchicalReasoningModel_ACTV1Carry(
            new_inner_carry, new_steps, halted, new_current_data
        ), outputs