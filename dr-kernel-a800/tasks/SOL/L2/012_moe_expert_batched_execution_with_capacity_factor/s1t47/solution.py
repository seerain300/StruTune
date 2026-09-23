import torch
import torch.nn as nn
import triton
import triton.language as tl


# Triton kernel: Y = X @ W, where X is 1xK (passed as a contiguous vector of length K),
# W is [K, M] row-major, Y is 1xM vector.
@triton.jit
def _matmul_1xK_KxM_kernel(X_ptr, W_ptr, Y_ptr,
                            K: tl.int32, M: tl.int32,
                            stride_xk, stride_wk, stride_wm, stride_ym,
                            BLOCK_M: tl.constexpr):
    acc = tl.zeros((M,), dtype=tl.float32)
    # Loop over K dimension
    for k in range(0, K):
        x_k = tl.load(X_ptr + k * stride_xk)
        w_row = tl.load(W_ptr + k * stride_wk + tl.arange(0, BLOCK_M) * stride_wm, mask=tl.arange(0, BLOCK_M) < M, other=0.0)
        acc += x_k * w_row
    # Store result into Y (contiguous 1D of length M)
    tl.store(Y_ptr + tl.arange(0, BLOCK_M) * stride_ym, acc, mask=tl.arange(0, BLOCK_M) < M)


# Triton elementwise: Y = silu(Z) = Z * sigmoid(Z)
@triton.jit
def _silu_kernel(Z_ptr, Y_ptr, N: tl.int32, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    z = tl.load(Z_ptr + offsets, mask=mask, other=0.0)  # fp32
    sig = 1.0 / (1.0 + tl.exp(-z))
    y = z * sig
    tl.store(Y_ptr + offsets, y, mask=mask)


# Triton elementwise: Y = A * B (same length N)
@triton.jit
def _mul_kernel(A_ptr, B_ptr, Y_ptr, N: tl.int32, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    y = a * b
    tl.store(Y_ptr + offsets, y, mask=mask)


# Triton atomic accumulation into OUT row (1D contiguous vector of length N)
# Adds ROW contribution (1D vector of length N) to OUT. OUT must be initialized to zeros.
@triton.jit
def _atomic_accumulate_row_kernel(ROW_ptr, OUT_ptr, N: tl.int32, stride_out, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    row_vals = tl.load(ROW_ptr + offsets, mask=mask, other=0.0)  # fp32
    # Atomic add into OUT
    tl.atomic_add(OUT_ptr + offsets * stride_out, row_vals, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        """
        hidden_states: [num_tokens, hidden_size], dtype=bfloat16, device=CUDA
        selected_experts: [num_tokens, num_experts_per_tok], int64
        routing_weights: [num_tokens, num_experts_per_tok], bfloat16
        expert_gate_weights: [num_experts, hidden_size, moe_intermediate_size]
        expert_up_weights:   [num_experts, hidden_size, moe_intermediate_size]
        expert_down_weights: [num_experts, moe_intermediate_size, hidden_size]
        Returns: [num_tokens, hidden_size], bfloat16
        """
        assert hidden_states.is_cuda and hidden_states.dtype == torch.bfloat16
        assert selected_experts.is_cuda and routing_weights.is_cuda
        # Shapes
        num_tokens, hidden_size = hidden_states.shape
        num_experts, H_in, H_out = expert_gate_weights.shape
        # We assume the provided selected_experts are valid. We will loop deterministically.
        # Prepare output result in fp32 for atomic accumulation; cast to bfloat16 at the end.
        result = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=hidden_states.device)

        # Constants for Triton kernel tiling
        BLOCK_M = 128  # for matmul and atomic accumulation blocks
        BLOCK_ELM = 256  # for elementwise kernels

        # Process each token
        for t in range(num_tokens):
            # Get selected experts for this token (int64 vector of length K)
            K = int(selected_experts.shape[1])
            exp_list = selected_experts[t].to(torch.int64)  # already int64
            # Iterate over K selected experts
            for j in range(K):
                e = int(exp_list[j].item())  # Triton requires int constants for indexing

                # 1) Compute gate_out = hidden_states[t] @ expert_gate_weights[e]
                #    Shape: hidden states -> [H_in], gate_weights -> [H_in, H_out]
                gate_out_fp32 = torch.empty(H_out, dtype=torch.float32, device=hidden_states.device)
                X = hidden_states[t]  # 1D fp32 load; we pass pointer directly
                W = expert_gate_weights[e]  # [H_in, H_out]
                # Launch Triton kernel: X is 1xH_in, W is [H_in, H_out]
                grid_mat = (triton.cdiv(H_out, BLOCK_M),)
                _matmul_1xK_KxM_kernel[grid_mat](
                    X, W, gate_out_fp32,
                    H_in, H_out,
                    1, 1, H_out, H_out,
                    BLOCK_M=BLOCK_M,
                )

                # 2) Compute up_out = hidden_states[t] @ expert_up_weights[e]
                up_out_fp32 = torch.empty(H_out, dtype=torch.float32, device=hidden_states.device)
                W2 = expert_up_weights[e]  # [H_in, H_out]
                _matmul_1xK_KxM_kernel[grid_mat](
                    X, W2, up_out_fp32,
                    H_in, H_out,
                    1, 1, H_out, H_out,
                    BLOCK_M=BLOCK_M,
                )

                # 3) activated = silu(gate_out) * up_out
                activated_fp32 = torch.empty(H_out, dtype=torch.float32, device=hidden_states.device)
                # silu
                _silu_kernel[(triton.cdiv(H_out, BLOCK_ELM),)](
                    gate_out_fp32, activated_fp32, H_out, BLOCK_ELM
                )
                # multiply
                _mul_kernel[(triton.cdiv(H_out, BLOCK_ELM),)](
                    activated_fp32, up_out_fp32, activated_fp32, H_out, BLOCK_ELM
                )

                # 4) final_out = activated @ expert_down_weights[e]
                #    activated: [H_out], down_weights: [H_out, H_in], final_out: [H_in]
                final_out_fp32 = torch.empty(hidden_size, dtype=torch.float32, device=hidden_states.device)
                W3 = expert_down_weights[e]  # [H_out, H_in]
                _matmul_1xK_KxM_kernel[(triton.cdiv(hidden_size, BLOCK_M),)](
                    activated_fp32, W3, final_out_fp32,
                    H_out, hidden_size,
                    1, H_out, 1, hidden_size,
                    BLOCK_M=BLOCK_M,
                )

                # 5) Accumulate: result[t] += routing_weights[t, j] * final_out
                weight = routing_weights[t, j].item()  # scalar float32
                contribution = final_out_fp32 * weight
                # Atomic add into result[t] (1D vector of length hidden_size)
                _atomic_accumulate_row_kernel[(triton.cdiv(hidden_size, BLOCK_ELM),)](
                    contribution, result[t], hidden_size, 1, BLOCK_ELM
                )

        # Cast result to bfloat16 for return
        return result.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
