import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


# Triton kernel: per-row matmul producing a single vector C[m, :] = A_row[m, :] @ B[:, :]
@triton.jit
def triton_row_matmul(C_ptr, A_row_ptr, B_ptr, N,  # N = number of columns in B (output vector length)
                       K,  # number of rows in A (and number of columns in A; equals N if square, otherwise general)
                       M,  # row index m
                       BLOCK: tl.constexpr):
    # Accumulator vector
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    offs = tl.arange(0, BLOCK)
    # Loop over K dimension in chunks of BLOCK
    for k0 in range(0, K, BLOCK):
        k_ids = k0 + offs
        # Load A_row chunk and B chunk
        a_chunk = tl.load(A_row_ptr + k_ids, mask=k_ids < K, other=0.0)  # A_row[m, k_ids]
        b_chunk = tl.load(B_ptr + k_ids * N + offs, mask=k_ids < K, other=0.0)  # B[k_ids, offs]
        # acc += sum_k a[m, k] * B[k, :]
        acc += tl.sum(a_chunk[:, None] * b_chunk[None, :], axis=0)
    # Store result to C[m, :]
    tl.store(C_ptr + M * N + offs, acc, mask=offs < N)


# Triton kernel: elementwise SiLU on a 1D vector
@triton.jit
def triton_silu(vec_ptr, out_ptr, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    idx = offs + tl.program_id(0) * BLOCK
    mask = idx < N
    x = tl.load(vec_ptr + idx, mask=mask, other=0.0)
    s = 1.0 / (1.0 + tl.exp(-x))
    y = x * s
    tl.store(out_ptr + idx, y, mask=mask)


# Triton kernel: elementwise multiply of two 1D vectors
@triton.jit
def triton_mul(vec_a_ptr, vec_b_ptr, out_ptr, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    idx = offs + tl.program_id(0) * BLOCK
    mask = idx < N
    a = tl.load(vec_a_ptr + idx, mask=mask, other=0.0)
    b = tl.load(vec_b_ptr + idx, mask=mask, other=0.0)
    tl.store(out_ptr + idx, a * b, mask=mask)


# Triton kernel: atomic add of a vector into out[t, :] at row index t
@triton.jit
def triton_atomic_add_row(out_ptr, vec_ptr, T, N, t, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    idx = offs + tl.program_id(0) * BLOCK
    mask = idx < N
    val = tl.load(vec_ptr + idx, mask=mask, other=0.0)
    old = tl.load(out_ptr + t * N + idx, mask=mask, other=0.0)
    new = old + val
    tl.store(out_ptr + t * N + idx, new, mask=mask)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_gate_weights: torch.Tensor,
    expert_up_weights: torch.Tensor,
    expert_down_weights: torch.Tensor,
):
    # Shapes
    num_tokens, hidden_size = hidden_states.shape
    num_experts, gate_K, _ = expert_gate_weights.shape  # gate K = hidden_size, output N = intermediate_size
    _, up_K, _ = expert_up_weights.shape
    _, down_K, _ = expert_down_weights.shape  # down K = intermediate_size, output N = hidden_size
    num_experts_per_tok = selected_experts.shape[1]

    device = hidden_states.device
    dtype = hidden_states.dtype  # bfloat16 in inputs

    # Output: float32 for numerical stability, will cast to bfloat16 at the end
    result = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=device)

    # Iterate tokens
    for t in range(num_tokens):
        # Iterate selected experts for this token
        for e in range(num_experts_per_tok):
            exp_id = int(selected_experts[t, e].item())

            # Compute gate_out = hidden_states[t, :] @ expert_gate_weights[exp_id, :, :]
            gate_out = None
            # Prepare inputs for Triton per-row matmul
            A_row_gate = hidden_states[t, :].contiguous().to(torch.float32)
            B_gate = expert_gate_weights[exp_id].contiguous().to(torch.float32)  # shape [hidden_size, intermediate_size]
            N_gate = B_gate.shape[1]  # intermediate_size
            K_gate = B_gate.shape[0]  # hidden_size
            # Allocate output vector for gate_out
            gate_out = torch.empty(N_gate, dtype=torch.float32, device=device)
            # Launch Triton per-row matmul kernel
            grid_gate = (1,)
            triton_row_matmul[grid_gate](gate_out, A_row_gate, B_gate, N_gate, K_gate, t, BLOCK=128)

            # Compute up_out = hidden_states[t, :] @ expert_up_weights[exp_id, :, :]
            up_out = None
            A_row_up = hidden_states[t, :].contiguous().to(torch.float32)
            B_up = expert_up_weights[exp_id].contiguous().to(torch.float32)  # shape [hidden_size, intermediate_size]
            N_up = B_up.shape[1]
            K_up = B_up.shape[0]
            up_out = torch.empty(N_up, dtype=torch.float32, device=device)
            grid_up = (1,)
            triton_row_matmul[grid_up](up_out, A_row_up, B_up, N_up, K_up, t, BLOCK=128)

            # SiLU(gate_out)
            gate_silu = torch.empty_like(gate_out, dtype=torch.float32, device=device)
            grid_silu = (triton.cdiv(N_gate, 256),)
            triton_silu[grid_silu](gate_out, gate_silu, N_gate, BLOCK=256)

            # Multiply SiLU(gate_out) with up_out (SwiGLU)
            activated = torch.empty_like(up_out, dtype=torch.float32, device=device)
            grid_mul = (triton.cdiv(N_up, 256),)
            triton_mul[grid_mul](gate_silu, up_out, activated, N_up, BLOCK=256)

            # Compute expert_outputs = activated @ expert_down_weights[exp_id, :, :]
            expert_out = None
            A_row_down = activated  # length = intermediate_size
            B_down = expert_down_weights[exp_id].contiguous().to(torch.float32)  # shape [intermediate_size, hidden_size]
            N_down = B_down.shape[1]  # hidden_size
            K_down = B_down.shape[0]  # intermediate_size
            expert_out = torch.empty(N_down, dtype=torch.float32, device=device)
            grid_down = (1,)
            triton_row_matmul[grid_down](expert_out, A_row_down, B_down, N_down, K_down, t, BLOCK=128)

            # Apply routing weight
            weight = float(routing_weights[t, e].item())
            expert_out = expert_out * weight

            # Atomically add to result[t, :]
            grid_add = (triton.cdiv(N_down, 256),)
            triton_atomic_add_row[grid_add](result, expert_out, num_tokens, hidden_size, t, BLOCK=256)

    # Cast to bfloat16 to match original dtype
    return result.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args expected: hidden_states, selected_experts, routing_weights, expert_gate_weights, expert_up_weights, expert_down_weights
        return run(*args)


def run(*args):
    return ModelNew()(*args)
