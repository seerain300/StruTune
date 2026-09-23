import math
import torch
import triton
import triton.language as tl


# Triton kernel: compute gate_out = A @ B, where A is a single row [hidden_size], B is [hidden_size, intermediate_size], output [intermediate_size].
@triton.jit
def bmm_row_gate_kernel(
    A_ptr, B_ptr, C_ptr,
    hidden_size: tl.constexpr, intermediate_size: tl.constexpr,
    m_offset: tl.constexpr,  # linear index of the row in flattened output
    BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr,
):
    offs_j = tl.arange(0, BLOCK_J)
    acc = tl.zeros((BLOCK_J,), dtype=tl.float32)
    base_a = m_offset * hidden_size
    for k0 in range(0, hidden_size, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_idx < hidden_size
        a = tl.load(A_ptr + base_a + k_idx, mask=mask_k, other=0.0)
        b = tl.load(B_ptr + k_idx[:, None] * intermediate_size + offs_j[None, :],
                    mask=(mask_k[:, None]), other=0.0)
        acc += tl.sum(b * a[:, None], axis=0)
    c_ptrs = C_ptr + m_offset * intermediate_size + offs_j
    tl.store(c_ptrs, acc, mask=(offs_j < intermediate_size))


# Triton kernel: compute up_out = A @ B, where A is a single row [hidden_size], B is [hidden_size, intermediate_size], output [intermediate_size].
@triton.jit
def bmm_row_up_kernel(
    A_ptr, B_ptr, C_ptr,
    hidden_size: tl.constexpr, intermediate_size: tl.constexpr,
    m_offset: tl.constexpr,
    BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr,
):
    offs_j = tl.arange(0, BLOCK_J)
    acc = tl.zeros((BLOCK_J,), dtype=tl.float32)
    base_a = m_offset * hidden_size
    for k0 in range(0, hidden_size, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_idx < hidden_size
        a = tl.load(A_ptr + base_a + k_idx, mask=mask_k, other=0.0)
        b = tl.load(B_ptr + k_idx[:, None] * intermediate_size + offs_j[None, :],
                    mask=(mask_k[:, None]), other=0.0)
        acc += tl.sum(b * a[:, None], axis=0)
    c_ptrs = C_ptr + m_offset * intermediate_size + offs_j
    tl.store(c_ptrs, acc, mask=(offs_j < intermediate_size))


# Triton kernel: compute expert_outputs_all = activated @ C, where activated is a single row [intermediate_size], C is [intermediate_size, hidden_size], output [hidden_size].
@triton.jit
def bmm_row_down_kernel(
    A_ptr, B_ptr, C_ptr,
    intermediate_size: tl.constexpr, hidden_size: tl.constexpr,
    m_offset: tl.constexpr,  # linear index of the row in flattened output
    BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr,
):
    offs_j = tl.arange(0, BLOCK_J)
    acc = tl.zeros((BLOCK_J,), dtype=tl.float32)
    # A is row vector of length intermediate_size
    base_a = m_offset * intermediate_size
    for k0 in range(0, intermediate_size, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_idx < intermediate_size
        a = tl.load(A_ptr + base_a + k_idx, mask=mask_k, other=0.0)
        b = tl.load(B_ptr + k_idx[:, None] * hidden_size + offs_j[None, :],
                    mask=(mask_k[:, None]), other=0.0)
        acc += tl.sum(b * a[:, None], axis=0)
    c_ptrs = C_ptr + m_offset * hidden_size + offs_j
    tl.store(c_ptrs, acc, mask=(offs_j < hidden_size))


# Triton kernel: scatter-add weighted rows into result.
# For each i in [0, N), add weighted_out[i, :] to result[v_tok[i], :]. Uses atomic add to handle duplicates.
@triton.jit
def scatter_add_rows_kernel(
    weights_ptr, pos_ptr, tok_ptr, a_ptr, result_ptr,
    N: tl.constexpr, hidden_size: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    i = tl.program_id(0)
    if i >= N:
        return
    w = tl.load(weights_ptr + i)
    pos = tl.load(pos_ptr + i).to(tl.int32)
    tok = tl.load(tok_ptr + i).to(tl.int32)
    offs_j = tl.arange(0, BLOCK_J)
    vals = tl.load(a_ptr + offs_j, mask=(offs_j < hidden_size), other=0.0)
    dst = result_ptr + tok * hidden_size + offs_j
    tl.atomic_add(dst, vals, mask=(offs_j < hidden_size))


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_gate_weights: torch.Tensor,
    expert_up_weights: torch.Tensor,
    expert_down_weights: torch.Tensor,
):
    # Ensure CUDA tensors
    assert hidden_states.is_cuda, "hidden_states must be CUDA tensor"
    device = hidden_states.device

    num_tokens, hidden_size = hidden_states.shape
    num_experts, _, intermediate_size = expert_gate_weights.shape
    num_experts_per_tok = selected_experts.shape[1]

    # Flatten assignment and weights
    flat_experts = selected_experts.reshape(-1)  # [T]
    flat_weights = routing_weights.reshape(-1)   # [T]
    T = flat_experts.numel()

    # capacity constraint: capacity = ceil(1.25 * T / num_experts), at least 1
    capacity = int(math.ceil(T * 1.25 / num_experts))
    capacity = max(1, capacity)

    # Compute counts and starts using torch (we only need to check validity; we will compute within_pos directly via argsort-based mapping)
    # Build sorted arrays
    perm = torch.randperm(T, device=device)
    sorted_experts = flat_experts[perm]
    sorted_weights = flat_weights[perm]
    sorted_token_ids = torch.arange(num_tokens, device=device).repeat_interleave(num_experts_per_tok)[perm]

    # For each original index i, compute within_pos = i - starts[selected_experts[i]]
    # We need starts: counts of experts in sorted list.
    counts = torch.bincount(sorted_experts)  # [num_experts]
    starts = torch.cumsum(counts, dim=0) - counts  # [num_experts]
    # Map sorted_experts back to original indices
    inv_perm = torch.argsort(perm)  # [T], so that inv_perm[j] gives original index of sorted element j
    original_experts = sorted_experts[inv_perm]  # [T]
    within_pos = inv_perm.to(torch.int32) - starts[original_experts].to(torch.int32)

    # Valid mask: within_pos < capacity
    valid = within_pos < capacity
    v_exp = original_experts[valid].to(torch.int32)       # [num_valid]
    v_pos = within_pos[valid].to(torch.int32)             # [num_valid] (not used except for logical check)
    v_tok = sorted_token_ids[perm][valid].to(torch.int64) # [num_valid], original tokens
    v_wt = sorted_weights[perm][valid].to(torch.float32)  # [num_valid], float32 for math

    # Compute M_total = number of valid assignments
    M_total = v_exp.numel()

    # Buffers for computed rows
    gate_out = torch.empty((M_total, intermediate_size), dtype=torch.float32, device=device)
    up_out = torch.empty((M_total, intermediate_size), dtype=torch.float32, device=device)
    activated = torch.empty_like(gate_out)  # we will fill elementwise; but simpler to compute and then use gate_out*up_out in Triton

    # Launch Triton kernels to compute gate_out and up_out for each valid assignment
    # One program per row m in [0, M_total)
    grid = (M_total,)
    BLOCK_K = 128 if hidden_size >= 128 else 64
    BLOCK_J_gate = 128 if intermediate_size >= 128 else 64

    for m in range(M_total):
        # A_row: hidden_state at v_tok[m]
        A_row = hidden_states[v_tok[m]].to(torch.float32)  # [hidden_size]
        B_gate = expert_gate_weights[v_exp[m]].contiguous()  # [hidden_size, intermediate_size]
        bmm_row_gate_kernel[grid](
            A_row, B_gate, gate_out,
            hidden_size, intermediate_size, m,
            BLOCK_J=BLOCK_J_gate, BLOCK_K=BLOCK_K,
        )
        B_up = expert_up_weights[v_exp[m]].contiguous()  # [hidden_size, intermediate_size]
        bmm_row_up_kernel[grid](
            A_row, B_up, up_out,
            hidden_size, intermediate_size, m,
            BLOCK_J=BLOCK_J_gate, BLOCK_K=BLOCK_K,
        )

    # Elementwise SiLU and multiply on PyTorch (small and correctness-critical)
    # SiLU(x) = x * sigmoid(x)
    activated = gate_out * torch.sigmoid(gate_out) * up_out  # [M_total, intermediate_size], float32

    # Compute expert_outputs_all = activated @ expert_down_weights per (e, n)
    # We need expert_down_weights per v_exp and activated rows. Use Triton row kernel
    expert_down = expert_down_weights[v_exp]  # [M_total, intermediate_size, hidden_size]
    expert_outputs_all = torch.empty((M_total, hidden_size), dtype=torch.float32, device=device)
    BLOCK_J_down = 128 if hidden_size >= 128 else 64
    for m in range(M_total):
        A_row = activated[m]  # [intermediate_size]
        B_down = expert_down[m]  # [intermediate_size, hidden_size]
        bmm_row_down_kernel[grid](
            A_row, B_down, expert_outputs_all,
            intermediate_size, hidden_size, m,
            BLOCK_J=BLOCK_J_down, BLOCK_K=intermediate_size,
        )

    # Scatter-add into result per token using Triton kernel (atomic_add ensures correctness for duplicates)
    result = torch.zeros((num_tokens, hidden_size), dtype=torch.float32, device=device)
    N = M_total
    grid_scatter = (N,)
    BLOCK_J_hs = 128 if hidden_size >= 128 else 64
    scatter_add_rows_kernel[grid_scatter](
        v_wt, v_pos, v_tok.to(torch.int32), expert_outputs_all, result,
        N, hidden_size, BLOCK_J=BLOCK_J_hs,
    )

    # Cast result back to bfloat16 to match original dtype
    result_bf16 = result.to(torch.bfloat16)
    return result_bf16


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        # Ensure CUDA tensors for Triton
        assert hidden_states.is_cuda and selected_experts.is_cuda and routing_weights.is_cuda and \
               expert_gate_weights.is_cuda and expert_up_weights.is_cuda and expert_down_weights.is_cuda, \
            "All tensors must be CUDA tensors for Triton execution."
        return run(hidden_states, selected_experts, routing_weights, expert_gate_weights, expert_up_weights, expert_down_weights)


def run(*args):
    return ModelNew()(*args)
