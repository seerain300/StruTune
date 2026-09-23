import math
import torch
import triton
import triton.language as tl


# Triton kernel: row-wise matmul for gate_out = hidden_row @ expert_gate_weights
# A: hidden_row (H,), B: expert_gate_weights (H, M), C: gate_out (M,)
@triton.jit
def row_bmm_gate(
    A_ptr,                 # *dtype, length H
    B_ptr,                 # *dtype, shape [H, M], row-major
    C_ptr,                 # *dtype, length M
    H: tl.int32,
    M: tl.int32,
    stride_b0: tl.int32,   # stride for dim 0 (H)
    stride_b1: tl.int32,   # stride for dim 1 (M)
    BLOCK_H: tl.constexpr, # tile size along H
    BLOCK_M: tl.constexpr, # tile size along M
):
    pid_m = tl.program_id(0)  # tile id along M
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # accumulator
    acc = tl.zeros([BLOCK_M], dtype=tl.dtype_of(A_ptr)  # use float16/float32 accordingly
                   )

    # loop over H
    for h_start in range(0, H, BLOCK_H):
        offs_h = h_start + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H

        # load A chunk: hidden_row[offs_h]
        a = tl.load(A_ptr + offs_h, mask=mask_h, other=0.0)

        # load B chunk: B[offs_h, offs_m] with masks
        b_ptrs = B_ptr + offs_h[:, None] * stride_b0 + offs_m[None, :] * stride_b1
        b = tl.load(b_ptrs, mask=mask_h[:, None] & mask_m[None, :], other=0.0)

        # dot: [BLOCK_H] x [BLOCK_H, BLOCK_M] -> [BLOCK_M]
        acc += tl.sum(b * a[:, None], axis=0)

    # store result
    tl.store(C_ptr + offs_m, acc, mask=mask_m)


# Triton kernel: row-wise matmul for up_out = hidden_row @ expert_up_weights
@triton.jit
def row_bmm_up(
    A_ptr,                 # *dtype, length H
    B_ptr,                 # *dtype, shape [H, M], row-major
    C_ptr,                 # *dtype, length M
    H: tl.int32,
    M: tl.int32,
    stride_b0: tl.int32,   # stride for dim 0 (H)
    stride_b1: tl.int32,   # stride for dim 1 (M)
    BLOCK_H: tl.constexpr, # tile size along H
    BLOCK_M: tl.constexpr, # tile size along M
):
    pid_m = tl.program_id(0)  # tile id along M
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    acc = tl.zeros([BLOCK_M], dtype=tl.dtype_of(A_ptr))

    for h_start in range(0, H, BLOCK_H):
        offs_h = h_start + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H

        a = tl.load(A_ptr + offs_h, mask=mask_h, other=0.0)

        b_ptrs = B_ptr + offs_h[:, None] * stride_b0 + offs_m[None, :] * stride_b1
        b = tl.load(b_ptrs, mask=mask_h[:, None] & mask_m[None, :], other=0.0)

        acc += tl.sum(b * a[:, None], axis=0)

    tl.store(C_ptr + offs_m, acc, mask=mask_m)


# Triton kernel: row-wise matmul for expert_outputs = activated @ expert_down_weights
@triton.jit
def row_bmm_down(
    A_ptr,                 # *dtype, length M
    B_ptr,                 # *dtype, shape [M, H], row-major
    C_ptr,                 # *dtype, length H
    M: tl.int32,
    H: tl.int32,
    stride_b0: tl.int32,   # stride for dim 0 (M)
    stride_b1: tl.int32,   # stride for dim 1 (H)
    BLOCK_M: tl.constexpr, # tile size along M
    BLOCK_H: tl.constexpr, # tile size along H
):
    pid_h = tl.program_id(0)  # tile id along H
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    acc = tl.zeros([BLOCK_H], dtype=tl.dtype_of(A_ptr))

    for m_start in range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        a = tl.load(A_ptr + offs_m, mask=mask_m, other=0.0)

        b_ptrs = B_ptr + offs_m[:, None] * stride_b0 + offs_h[None, :] * stride_b1
        b = tl.load(b_ptrs, mask=mask_m[:, None] & mask_h[None, :], other=0.0)

        acc += tl.sum(b * a[:, None], axis=0)

    tl.store(C_ptr + offs_h, acc, mask=mask_h)


# Triton kernel: SiLU elementwise y = x * sigmoid(x)
@triton.jit
def silu_kernel(
    x_ptr,                 # *dtype, length N
    y_ptr,                 # *dtype, length N
    N: tl.int32,
):
    pid = tl.program_id(0)
    if pid < N:
        x = tl.load(x_ptr + pid)
        # sigmoid
        s = 1.0 / (1.0 + tl.exp(-x))
        y = x * s
        tl.store(y_ptr + pid, y)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Extract inputs from args: (hidden_states, selected_experts, routing_weights,
        # expert_gate_weights, expert_up_weights, expert_down_weights)
        hidden_states = args[0]  # [T, H], bfloat16
        selected_experts = args[1]  # [T, K], int64
        routing_weights = args[2]  # [T, K], dtype
        expert_gate_weights = args[3]  # [E, H, M], dtype
        expert_up_weights = args[4]  # [E, H, M], dtype
        expert_down_weights = args[5]  # [E, M, H], dtype

        T = hidden_states.shape[0]
        H = hidden_states.shape[1]
        E = expert_gate_weights.shape[0]
        M = expert_gate_weights.shape[2]
        K = selected_experts.shape[1]

        # Flatten selected_experts and routing_weights (host-side)
        flat_token_exp = selected_experts.reshape(-1)          # int64 [T*K]
        flat_weights = routing_weights.reshape(-1)             # dtype [T*K]
        flat_token_ids = torch.arange(T, device=hidden_states.device).repeat_interleave(K)  # int64 [T*K]

        # Stable sort by expert id. Use torch for robustness.
        # Note: chosen_experts_flat is not used directly; we sort flat_token_exp and weights together.
        _, perm = torch.sort(flat_token_exp, stable=True)
        sorted_experts = flat_token_exp[perm].to(torch.int64)
        sorted_weights = flat_weights[perm]
        sorted_token_ids = flat_token_ids[perm]

        # Compute capacity per expert
        capacity_scalar = (T * K * 4) // (E * 5)  # int((T*K)/1.25)
        capacity_scalar = max(capacity_scalar, 1)
        capacity = int(capacity_scalar)

        # Build valid mask: after stable sort, tokens per expert are contiguous.
        counts = torch.bincount(sorted_experts.cpu(), minlength=E)  # CPU bincount is fine
        # Compute starts on device: starts[i] = sum_j counts[j] for j < i
        # We can compute this with torch.cumsum and send back to device.
        starts_cpu = torch.zeros(E, dtype=torch.int32, device='cpu')
        if E > 0:
            starts_cpu[1:] = counts[:-1].cumsum(dim=0)
        starts = starts_cpu.to(device=hidden_states.device, dtype=torch.int32)

        # Compute global indices (sorted order): idx = position in sorted array
        idx_long = torch.arange(T * K, device=hidden_states.device, dtype=torch.int64)
        within_pos = idx_long - starts[sorted_experts]

        valid = within_pos < capacity
        v_exp = sorted_experts[valid]              # int64 [Nv]
        v_tok = sorted_token_ids[valid]           # int64 [Nv]
        v_weight = sorted_weights[valid]          # dtype [Nv]
        v_pos = within_pos[valid].to(torch.int32) # int32 [Nv]

        # Build padded expert_inputs [E, capacity, H]
        # expert_inputs[exp, pos, :] = hidden_states[token_ids[v_tok]] when valid
        expert_inputs = torch.zeros(E, capacity, H, device=hidden_states.device, dtype=hidden_states.dtype)

        # For invalid positions (if capacity < counts), set inputs to 0; here we only scatter valid ones.
        # Gather rows: hidden_states[v_tok, :] and place at expert_inputs[v_exp, v_pos, :].
        # Scatter is supported via advanced indexing in PyTorch.
        expert_inputs[v_exp, v_pos] = hidden_states[v_tok]

        # Now perform Triton kernels for batched matmuls. We iterate over valid pairs and launch kernels.
        # Prepare output buffers for each pair.
        # We'll create intermediate gate_out, up_out, activated, expert_outputs for each pair and aggregate.
        # However, launching per-pair Triton calls is cumbersome in Python. Instead, we implement one row_bmm kernel
        # for a single row and call it multiple times; Triton JIT compiles per signature, which is fine for moderate Nv.
        # For robustness, use small BLOCK sizes (e.g., 64/128).
        BLOCK_H = 128
        BLOCK_M = 64
        output_per_token = torch.zeros(T, H, device=hidden_states.device, dtype=hidden_states.dtype)

        # For each valid pair, compute gate_out, up_out, activated, expert_outputs, and accumulate into output_per_token.
        # We need to map v_tok to output_per_token rows. However, multiple pairs can map to the same token.
        # So we accumulate: index_add per token.

        # Implement a loop over Nv; Triton will compile per signature. This is acceptable for the given workload sizes.
        # We'll iterate and launch kernels. Triton requires grid to be tuple. Here, M dimension controls grid size.

        # For clarity, we can structure as nested loops; Triton JIT handles repeated kernels. But to keep it simple,
        # we will launch kernels by varying a "pair_id" index and using scalar inputs. Triton supports scalar args.

        # To avoid mismatch, we'll use Triton kernels with grid dependent on M. We can set grid as (ceil_div(M, BLOCK_M),).
        # However, since Nv may vary, we'll iterate and launch per pair with a fixed grid computed from M/H.

        # Helper to compute grid: lambda H or M
        def grid_bmm(H_val, M_val, BLOCK_H_val, BLOCK_M_val):
            return (triton.cdiv(M_val, BLOCK_M_val),)

        # Loop over valid pairs
        for p in range(0, valid.sum().item()):
            exp = int(v_exp[p].item())
            pos = int(v_pos[p].item())
            tok = int(v_tok[p].item())

            # Gate Out: hidden_states[tok] @ expert_gate_weights[exp]
            hidden_row = hidden_states[tok]  # [H], dtype
            # expert_gate_weights[exp] is [H, M], we need to pass as pointer. Triton expects strides.
            gate_out = torch.empty(M, device=hidden_states.device, dtype=hidden_states.dtype)
            # Launch kernel for gate_out
            row_bmm_gate[grid_bmm(H, M, BLOCK_H, BLOCK_M)](
                hidden_row,                # A_ptr
                expert_gate_weights[exp],  # B_ptr: [H, M]
                gate_out,                  # C_ptr: [M]
                H, M,
                M * 1,                     # stride_b0 (row stride in elements for [H, M])
                1,                         # stride_b1 (col stride in elements for [H, M])
                BLOCK_H=BLOCK_H,
                BLOCK_M=BLOCK_M,
            )

            # Up Out: hidden_states[tok] @ expert_up_weights[exp]
            up_out = torch.empty(M, device=hidden_states.device, dtype=hidden_states.dtype)
            row_bmm_up[grid_bmm(H, M, BLOCK_H, BLOCK_M)](
                hidden_row,
                expert_up_weights[exp],  # [H, M]
                up_out,
                H, M,
                M * 1,
                1,
                BLOCK_H=BLOCK_H,
                BLOCK_M=BLOCK_M,
            )

            # SiLU and elementwise mul
            silu_out = torch.empty(M, device=hidden_states.device, dtype=hidden_states.dtype)
            silu_kernel[(M,)](
                gate_out,
                silu_out,
                M,
            )
            activated = silu_out * up_out  # elementwise, PyTorch for simplicity

            # Down: activated @ expert_down_weights[exp]
            # activated is [M], expert_down_weights[exp] is [M, H]
            expert_outputs = torch.empty(H, device=hidden_states.device, dtype=hidden_states.dtype)
            row_bmm_down[grid_bmm(M, H, BLOCK_M, BLOCK_H)](
                activated,                 # A_ptr: [M]
                expert_down_weights[exp],  # B_ptr: [M, H]
                expert_outputs,            # C_ptr: [H]
                M, H,
                H * 1,                     # stride_b0
                1,                         # stride_b1
                BLOCK_M=BLOCK_M,
                BLOCK_H=BLOCK_H,
            )

            # Accumulate into output_per_token[tok] += expert_outputs * v_weight[p]
            # We need to index_add result by tok. Create a vector of length H for tok.
            # But multiple pairs can map to same tok; so do index_add.
            out_vec = expert_outputs * float(v_weight[p].item())
            output_per_token.index_add_(0, tok, out_vec)

        # Return result (shape [T, H])
        return output_per_token


def run(*args):
    return ModelNew()(*args)
