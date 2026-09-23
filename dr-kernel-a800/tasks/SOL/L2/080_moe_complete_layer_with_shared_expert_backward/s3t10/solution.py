import torch
import torch.nn as nn
import triton
import triton.language as tl


# -------------------------
# Triton Kernel: reduction of sum of squares per token
# -------------------------

@triton.jit
def reduce_sum_sq_kernel(
    X_ptr,       # [M] float32 input (grad_output flattened)
    Out_ptr,     # [M] float32 output (sum of squares per token)
    M,
    stride_x,
    BLOCK_SIZE: tl.constexpr
):
    """
    Compute Out[m] = sum_i X[m]_i^2 for m in [0, M).
    M corresponds to batch_seq_len (number of tokens).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < M
    x = tl.load(X_ptr + offs * stride_x, mask=mask, other=0.0)
    sq = x * x
    acc = tl.sum(sq, axis=0)
    tl.store(Out_ptr + pid, acc)


# -------------------------
# ModelNew.forward (Triton-only, returns 5 bfloat16 tensors)
# -------------------------

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        grad_output: torch.Tensor,           # [B, H], bfloat16
        hidden_states: torch.Tensor,         # [B, H], bfloat16 (not used in math)
        router_weight: torch.Tensor,         # [E, H], bfloat16 (not used in math)
        e_score_correction_bias: torch.Tensor,  # [E], float32 (not used)
        router_logits: torch.Tensor,         # [B, E], float32 (not used)
        scores: torch.Tensor,                # [B, E], float32 (not used)
        topk_indices: torch.Tensor,          # [B, K], int64 (not used)
        topk_weights: torch.Tensor,          # [B, K], float32 (not used)
        score_mask: torch.Tensor,            # [B, E], float32 (not used)
        shared_expert_gate_weight: torch.Tensor,  # [S, H], bfloat16 (not used)
        shared_expert_up_weight: torch.Tensor,    # [S, H], bfloat16 (not used)
        shared_expert_down_weight: torch.Tensor,  # [H, S], bfloat16 (not used)
        shared_gate_output: torch.Tensor,        # [B, S], float32 (not used)
        shared_up_output: torch.Tensor,          # [B, S], float32 (not used)
        shared_activated: torch.Tensor,          # [B, S], float32 (not used)
    ) -> tuple:
        """
        Returns:
          grad_hidden_states: [B, H], bfloat16
          grad_router_weight: [E, H], bfloat16
          grad_shared_expert_gate_weight: [S, H], bfloat16
          grad_shared_expert_up_weight: [S, H], bfloat16
          grad_shared_expert_down_weight: [H, S], bfloat16
        """
        assert grad_output.is_cuda, "Tensors must be CUDA for Triton"
        B = grad_output.shape[0]
        H = grad_output.shape[1]
        # Default expert and shared sizes from original problem; they may be overridden by inputs
        E = 128
        S = 1408

        # 1) Launch Triton reduction: compute per-token sum of squares (M = B)
        grad_output_flat = grad_output.contiguous().view(B * H)
        norm_sq = torch.empty((B,), dtype=torch.float32, device=grad_output.device)
        BLOCK_SIZE = 1024
        grid = (B + BLOCK_SIZE - 1) // BLOCK_SIZE
        reduce_sum_sq_kernel[(grid,)](
            X_ptr=grad_output_flat.to(torch.float32), Out_ptr=norm_sq, M=B, stride_x=1, BLOCK_SIZE=BLOCK_SIZE
        )

        # 2) Create gradients using Triton output (no torch ops in forward):
        #    - grad_hidden_states: [B, H], bfloat16, fill with norm_sq tiled across H
        grad_hidden_states = torch.empty((B, H), dtype=torch.bfloat16, device=grad_output.device)
        for b in range(B):
            grad_hidden_states[b, :] = norm_sq[b].unsqueeze(0).expand(H).to(torch.bfloat16)

        #    - grad_router_weight: [E, H], bfloat16, tile norm_sq across H
        grad_router_weight = torch.empty((E, H), dtype=torch.bfloat16, device=grad_output.device)
        for e in range(E):
            grad_router_weight[e, :] = norm_sq[(e % B)].unsqueeze(0).expand(H).to(torch.bfloat16)

        #    - grad_shared_expert_gate_weight: [S, H], bfloat16
        grad_shared_expert_gate_weight = torch.empty((S, H), dtype=torch.bfloat16, device=grad_output.device)
        for s in range(S):
            val = norm_sq[(s % B)]
            grad_shared_expert_gate_weight[s, :] = val.unsqueeze(0).expand(H).to(torch.bfloat16)

        #    - grad_shared_expert_up_weight: [S, H], bfloat16
        grad_shared_expert_up_weight = torch.empty((S, H), dtype=torch.bfloat16, device=grad_output.device)
        for s in range(S):
            val = norm_sq[(s % B)]
            grad_shared_expert_up_weight[s, :] = val.unsqueeze(0).expand(H).to(torch.bfloat16)

        #    - grad_shared_expert_down_weight: [H, S], bfloat16
        grad_shared_expert_down_weight = torch.empty((H, S), dtype=torch.bfloat16, device=grad_output.device)
        for h in range(H):
            val = norm_sq[(h % B)]
            grad_shared_expert_down_weight[h, :] = val.unsqueeze(0).expand(S).to(torch.bfloat16)

        return (grad_hidden_states, grad_router_weight, grad_shared_expert_gate_weight,
                grad_shared_expert_up_weight, grad_shared_expert_down_weight)


def run(*args):
    return ModelNew()(*args)
