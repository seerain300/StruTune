import math
import torch
import triton
import triton.language as tl


@triton.jit
def triton_silu(x_ptr,          # [H], bfloat16
                out_ptr,         # [H], bfloat16
                H: tl.constexpr):
    for i in range(0, H):
        x = tl.load(x_ptr + i).to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-x))
        y = x * sig
        tl.store(out_ptr + i, y.to(tl.bfloat16))


@triton.jit
def triton_mul(a_ptr, b_ptr, out_ptr, H: tl.constexpr):
    for i in range(0, H):
        a = tl.load(a_ptr + i).to(tl.float32)
        b = tl.load(b_ptr + i).to(tl.float32)
        tl.store(out_ptr + i, (a * b).to(tl.bfloat16))


@triton.jit
def triton_atomic_weighted_add(weight_ptr,  # [N], bfloat16
                               vec_ptr,     # [N], bfloat16
                               out_ptr,     # [num_tokens, hidden_size], bfloat16
                               num_tokens: tl.constexpr,
                               hidden_size: tl.constexpr):
    # Atomic add weight * vec into out (flattened indexing)
    for i in range(0, num_tokens * hidden_size):
        weight = tl.load(weight_ptr + i).to(tl.float32)
        vec = tl.load(vec_ptr + i).to(tl.float32)
        out = tl.load(out_ptr + i)
        out += weight * vec
        tl.store(out_ptr + i, out)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        # Input shapes
        num_tokens, hidden_size = hidden_states.shape
        num_experts, _, moe_intermediate_size = expert_gate_weights.shape

        # Preprocess with torch (host-side) to match original semantics:
        # 1) Sort


def run(*args):
    return ModelNew()(*args)
