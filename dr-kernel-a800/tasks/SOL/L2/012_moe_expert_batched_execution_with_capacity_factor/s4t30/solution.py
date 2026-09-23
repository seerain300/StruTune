import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


# 2D Triton matmul kernel: C[M, N] = A[M, K] @ B[K, N]
@triton.jit
def triton_matmul_2d(C_ptr, A_ptr, B_ptr,
                     M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * K) + offs_k[None, :]
        b_ptrs = B_ptr + (offs_k[:, None] * K) + offs_n[None, :]
        A_tile = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        B_tile = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(A_tile, B_tile)
    c_ptrs = C_ptr + (offs_m[:, None] * N) + offs_n[None, :]
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Triton elementwise SiLU: y = x * sigmoid(x)
@triton.jit
def triton_silu_elementwise_vec(Y_ptr, X_ptr, L: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < L
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    y = x * tl.sigmoid(x)
    tl.store(Y_ptr + offsets, y, mask=mask)


# Triton elementwise multiply: Z = A * B
@triton.jit
def triton_mul_elementwise_vec(Z_ptr, A_ptr, B_ptr, L: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < L
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    tl.store(Z_ptr + offsets, a * b, mask=mask)


# Triton atomic add for weighted vector contributions into output
# We assume routing_weights are float32 and output is float32.
@triton.jit
def triton_atomic_add_weighted_vec(Out_ptr, In_ptr, Weights_ptr, L: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < L
    in_val = tl.load(In_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(Weights_ptr + offsets, mask=mask, other=0.0)
    contrib = in_val * weight
    # Atomic add per element
    tl.atomic_add(Out_ptr + offsets, contrib, mask=mask)


def _ceil_div(x, y):
    return (x + y - 1) // y


def _triton_matmul_2d(A, B, M, N, K, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64):
    # A: [M, K], B: [K, N], C: [M, N]
    grid = (_ceil_div(M, BLOCK_M), _ceil_div(N, BLOCK_N))
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    triton_matmul_2d[grid](C, A, B, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps=4, num_stages=3)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        # All heavy numerical compute in Triton; torch only for setup and minimal aggregation.
        num_tokens, hidden_size = hidden_states.shape
        num_experts, gate_h, gate_m = expert_gate_weights.shape
        num_experts_per_tok = selected_experts.shape[1]

        # Flatten token-expert selections
        flat_experts = selected_experts.reshape(-1)         # [num_tokens * K]
        flat_routing = routing_weights.reshape(-1)          # [num_tokens * K]

        # Precompute outputs and intermediates in float32 for accumulation
        # We will compute gate_out, up_out, activated, expert_outputs per token-expert pair
        # and then do Triton atomic adds into final_result per token.

        final_result = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=hidden_states.device)

        # Process each token and each selected expert
        for t in range(num_tokens):
            for j in range(num_experts_per_tok):
                exp = int(flat_experts[t * num_experts_per_tok + j])

                # Gate GEMM: gate_out[t*K + j] = hidden[t] @ expert_gate_weights[exp]
                # hidden[t]: [hidden_size], expert_gate_weights[exp]: [hidden_size, gate_m]
                # Treat as A[M=hidden_size, K=gate_m], B[K=gate_m, N=hidden_size]
                A_gate = hidden_states[t].to(torch.float32)  # [hidden_size]
                # Reshape to [hidden_size, gate_m] and compute with Triton
                A_gate = A_gate.view(hidden_size, gate_m)    # [M, K]
                B_gate = expert_gate_weights[exp].to(torch.float32).view(gate_m, hidden_size)  # [K, N]
                gate_out_vec = _triton_matmul_2d(A_gate, B_gate, hidden_size, hidden_size, gate_m)

                # Up GEMM: up_out[t*K + j] = hidden[t] @ expert_up_weights[exp]
                A_up = A_gate
                B_up = expert_up_weights[exp].to(torch.float32).view(gate_m, hidden_size)  # [K, N]
                up_out_vec = _triton_matmul_2d(A_up, B_up, hidden_size, hidden_size, gate_m)

                # SiLU on gate_out
                activated_vec = torch.empty(hidden_size, dtype=torch.float32, device=hidden_states.device)
                grid_silu = (_ceil_div(hidden_size, 256),)
                triton_silu_elementwise_vec[grid_silu](activated_vec, gate_out_vec, hidden_size, 256)

                # SwiGLU elementwise: activated * up_out
                fused_vec = torch.empty(hidden_size, dtype=torch.float32, device=hidden_states.device)
                grid_mul = (_ceil_div(hidden_size, 256),)
                triton_mul_elementwise_vec[grid_mul](fused_vec, activated_vec, up_out_vec, hidden_size, 256)

                # Down GEMM: expert_outputs[t*K + j] = fused_vec @ expert_down_weights[exp]
                # fused_vec: [hidden_size] -> A[M=hidden_size, K=intermediate], B[K=intermediate, N=hidden_size]
                # gate_m == hidden_size here? No, gate_m is the intermediate size.
                # We need to use the intermediate dimension for down GEMM.
                # Identify intermediate size from gate_m (moe_intermediate_size)
                intermediate = gate_m
                A_down = fused_vec.view(hidden_size, intermediate)  # [M, K]
                B_down = expert_down_weights[exp].to(torch.float32).view(intermediate, hidden_size)  # [K, N]
                expert_out_vec = _triton_matmul_2d(A_down, B_down, hidden_size, hidden_size, intermediate)

                # Atomic add weighted contribution into final_result[t]
                weight = flat_routing[t * num_experts_per_tok + j]  # scalar
                contrib_vec = expert_out_vec * weight
                # Use a vector of size hidden_size with same weight for atomic_add
                weights_vec = contrib_vec  # [hidden_size] filled with weight
                grid_atomic = (_ceil_div(hidden_size, 256),)
                triton_atomic_add_weighted_vec[grid_atomic](final_result[t], contrib_vec, weights_vec, hidden_size, 256)

        # Return bfloat16 as in original code
        return final_result.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
