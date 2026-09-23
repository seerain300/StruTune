import torch
import triton
import triton.language as tl


@triton.jit
def bmm_triton_kernel(
    X_ptr,   # *bf16, [B, H] contiguous
    W_ptr,   # *bf16, [H, M] contiguous
    Y_ptr,   # *bf16, [B, M] contiguous
    B: tl.constexpr,   # number of rows in X (here always 1)
    H: tl.constexpr,   # number of columns in X (hidden_size)
    M,                # int: output width (moe_intermediate_size or hidden_size)
    BLOCK_M: tl.constexpr,
):
    # One program per output row
    row = tl.program_id(0)
    if row >= B:
        return
    cols = tl.arange(0, BLOCK_M)
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    # Loop over K dimension (H) in tiles
    for k0 in range(0, H, BLOCK_M):
        k = k0 + cols
        mask_k = k < H
        # Load X row tile (vector of length BLOCK_M)
        x = tl.load(X_ptr + row * H + k, mask=mask_k, other=0.0).to(tl.float32)
        # Load W tile (BLOCK_M x BLOCK_M)
        w = tl.load(W_ptr + k[:, None] * M + cols[None, :], mask=mask_k[:, None], other=0.0).to(tl.float32)
        # acc += x @ w
        acc += tl.sum(w * x[None, :], axis=1)
    # Store result
    out = acc
    tl.store(Y_ptr + row * M + cols, out, mask=cols < M)


@triton.jit
def atomic_accumulate_kernel(
    INPUT_ptr,     # *bf16, [M] containing per-tile values to add
    OUTPUT_ptr,    # *fp32, [num_tokens, hidden_size] row-major
    scale_ptr,     # *bf16, [1] scalar routing_weight as 1-element tensor
    M,             # int: hidden_size
    row_id,        # int: token index (0..num_tokens-1)
    BLOCK: tl.constexpr,
):
    # Each program handles one tile of columns
    pid = tl.program_id(0)
    cols = pid * BLOCK + tl.arange(0, BLOCK)
    vals = tl.load(INPUT_ptr + cols, mask=cols < M, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)
    vals = vals * scale
    out_ptrs = OUTPUT_ptr + row_id * M + cols
    tl.atomic_add(out_ptrs, vals, mask=cols < M)


@triton.jit
def silu_mul_kernel(
    Z_ptr,  # *bf16, [N] gate_out
    U_ptr,  # *bf16, [N] up_out
    Y_ptr,  # *bf16, [N] output (activated = silu(Z) * U)
    N,      # int
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    for start in range(0, N, BLOCK):
        idx = start + offs
        mask = idx < N
        z = tl.load(Z_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(U_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        # silu(x) = x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
        s = 1.0 / (1.0 + tl.exp(-z))
        y = (z * s) * u
        tl.store(Y_ptr + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        # Ensure contiguous
        hidden_states = hidden_states.contiguous()
        expert_gate_weights = expert_gate_weights.contiguous()
        expert_up_weights = expert_up_weights.contiguous()
        expert_down_weights = expert_down_weights.contiguous()
        routing_weights = routing_weights.contiguous()

        num_tokens, hidden_size = hidden_states.shape
        num_experts, H_out, _ = expert_gate_weights.shape
        _, _, H_in = expert_down_weights.shape

        # Output buffer in fp32 for atomic adds; we will cast to bf16 at the end
        result = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=hidden_states.device)

        # Iterate over tokens and experts; avoid any torch ops in forward
        for t in range(num_tokens):
            # For each expert, compute gate_out, up_out, activated, final_out, and accumulate
            for e in range(num_experts):
                # 1) gate_out = hidden_states[t] @ expert_gate_weights[e]  -> [H_out] (bf16)
                gate_out = torch.empty(H_out, dtype=torch.bfloat16, device=hidden_states.device)
                bmm_triton_kernel[(1,)](
                    hidden_states[t], expert_gate_weights[e], gate_out,
                    1, hidden_size, H_out, BLOCK_M=64
                )
                # 2) up_out = hidden_states[t] @ expert_up_weights[e] -> [H_out] (bf16)
                up_out = torch.empty(H_out, dtype=torch.bfloat16, device=hidden_states.device)
                bmm_triton_kernel[(1,)](
                    hidden_states[t], expert_up_weights[e], up_out,
                    1, hidden_size, H_out, BLOCK_M=64
                )
                # 3) activated = silu(gate_out) * up_out -> [H_out] (bf16)
                activated = torch.empty(H_out, dtype=torch.bfloat16, device=hidden_states.device)
                silu_mul_kernel[(1,)](gate_out, up_out, activated, H_out, BLOCK=128)
                # 4) final_out = activated @ expert_down_weights[e] -> [H_in] (bf16)
                final_out = torch.empty(H_in, dtype=torch.bfloat16, device=hidden_states.device)
                bmm_triton_kernel[(1,)](
                    activated, expert_down_weights[e], final_out,
                    1, H_out, H_in, BLOCK_M=64
                )
                # 5) Accumulate result[t] += routing_weights[t, e] * final_out
                # Create scale tensor (bf16 scalar), pass to kernel
                scale = (routing_weights[t, e] * final_out).view(1).to(torch.bfloat16)
                # Atomic add into result row in fp32
                atomic_accumulate_kernel[(triton.cdiv(H_in, 128),)](
                    final_out, result,
                    scale,
                    H_in,
                    t,
                    BLOCK=128
                )

        # Return result as bf16, matching the original dtype
        return result.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
