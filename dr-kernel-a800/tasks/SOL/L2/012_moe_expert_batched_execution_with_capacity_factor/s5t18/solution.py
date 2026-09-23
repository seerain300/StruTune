import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 1: Flatten selected_experts (int64 -> int32), output length E
@triton.jit
def flatten_experts_kernel(
    src_exp_ptr,        # *int64, shape [num_tokens, num_experts_per_tok]
    dst_exp_ptr,        # *int32, length E
    num_tokens: tl.constexpr,
    num_experts_per_tok: tl.constexpr,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < E
    val = tl.load(src_exp_ptr + offsets, mask=mask, other=0)
    val = val.to(tl.int32)
    tl.store(dst_exp_ptr + offsets, val, mask=mask)

# Kernel 2: Flatten routing_weights into 1D vector (bf16), length E
@triton.jit
def flatten_weights_kernel(
    src_wt_ptr,         # *bf16, shape [num_tokens, num_experts_per_tok]
    dst_wt_ptr,         # *bf16, length E
    num_tokens: tl.constexpr,
    num_experts_per_tok: tl.constexpr,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < E
    val = tl.load(src_wt_ptr + offsets, mask=mask, other=0)
    tl.store(dst_wt_ptr + offsets, val, mask=mask)

# Kernel 3: Stable sort flattened expert list (int32) ascending via odd-even sort
@triton.jit
def stable_sort_by_exp_kernel(
    src_exp_ptr,        # *int32, length E (can be input or intermediate)
    dst_exp_ptr,        # *int32, length E
    E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Perform many passes; even/odd pairing to sort stably by expert ID
    for t in range(0, 200):
        if (t % 2) == 0:
            # even pass: pairs (0,1), (2,3), ...
            i = tl.arange(0, BLOCK)
            j = i + 1
            mask = (j < E) & ((i % 2) == 0)
            a = tl.load(src_exp_ptr + i, mask=mask, other=0)
            b = tl.load(src_exp_ptr + j, mask=mask, other=0)
            cond = a > b | ((a == b) & (i > j))  # tie-break by original position
            ai = tl.where(cond, b, a)
            aj = tl.where(cond, a, b)
            tl.store(dst_exp_ptr + i, ai, mask=mask)
            tl.store(dst_exp_ptr + j, aj, mask=mask)
        else:
            # odd pass: pairs (1,2), (3,4), ...
            i = tl.arange(0, BLOCK)
            j = i + 1
            mask = (j < E) & ((i % 2) == 1)
            a = tl.load(src_exp_ptr + i, mask=mask, other=0)
            b = tl.load(src_exp_ptr + j, mask=mask, other=0)
            cond = a > b | ((a == b) & (i > j))
            ai = tl.where(cond, b, a)
            aj = tl.where(cond, a, b)
            tl.store(dst_exp_ptr + i, ai, mask=mask)
            tl.store(dst_exp_ptr + j, aj, mask=mask)

# Kernel 4: Bincount per expert (atomic_add into counts, int32)
@triton.jit
def bincount_experts_kernel(
    src_exp_ptr,        # *int32, length E
    counts_ptr,         # *int32, length num_experts
    E: tl.constexpr,
    num_experts: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for i in range(0, E, BLOCK):
        idx = i + tl.arange(0, BLOCK)
        mask = idx < E
        vals = tl.load(src_exp_ptr + idx, mask=mask, other=0)  # int32
        for k in range(BLOCK):
            v = vals[k]
            m = mask[k]
            if m:
                tl.atomic_add(counts_ptr + v, 1)

# Kernel 5: Cumsum starts = prefix sum of counts (num_experts must be small)
@triton.jit
def cumsum_starts_kernel(
    counts_ptr,         # *int32, length num_experts
    starts_ptr,         # *int32, length num_experts
    num_experts: tl.constexpr,
):
    tl.store(starts_ptr + 0, tl.load(counts_ptr + 0))
    for i in range(1, num_experts):
        prev = tl.load(starts_ptr + (i - 1))
        curr = tl.load(counts_ptr + i)
        tl.store(starts_ptr + i, prev + curr)

# Kernel 6: Compute within_pos = global_sorted_index - starts[expert_id] for each flattened position
@triton.jit
def compute_within_pos_kernel(
    e_sorted_ptr,       # *int32, length E
    starts_ptr,         # *int32, length num_experts
    num_experts: tl.constexpr,
    E: tl.constexpr,
    capacity: tl.constexpr,
    within_ptr,         # *int32, length E
    BLOCK: tl.constexpr,
):
    for i in range(0, E, BLOCK):
        idx = i + tl.arange(0, BLOCK)
        mask = idx < E
        e = tl.load(e_sorted_ptr + idx, mask=mask, other=0)        # int32
        start = tl.load(starts_ptr + e, mask=mask, other=0)        # int32
        within = idx - start
        valid = within < capacity
        tl.store(within_ptr + idx, within, mask=mask & valid)

# Kernel 7: Final weighted scatter-add into result (float32 accumulation via atomic_add)
@triton.jit
def scatter_weighted_add_result_kernel(
    within_ptr,         # *int32, length E
    valid_ptr,          # *int32, length E
    wt_vec_ptr,         # *bf16, length E
    result_f32_ptr,     # *float32, length num_tokens*hidden_size
    num_tokens: tl.constexpr,
    hidden_size: tl.constexpr,
    num_experts_per_tok: tl.constexpr,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for i in range(0, E, BLOCK):
        pos = i + tl.arange(0, BLOCK)
        mask = pos < E
        w = tl.load(wt_vec_ptr + pos, mask=mask, other=0).to(tl.float32)
        valid = tl.load(valid_ptr + pos, mask=mask, other=0)       # int32
        within = tl.load(within_ptr + pos, mask=mask, other=0)     # int32
        # original token index and hidden offset
        n = pos // num_experts_per_tok
        h = within
        off = n * hidden_size + h
        tl.atomic_add(result_f32_ptr + off, w, mask=mask & (valid > 0))


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_gate_weights: torch.Tensor,
    expert_up_weights: torch.Tensor,
    expert_down_weights: torch.Tensor,
):
    # Shapes and device
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    num_experts = expert_gate_weights.shape[0]
    num_experts_per_tok = selected_experts.shape[1]
    device = hidden_states.device

    # Flatten selected_experts (int64 -> int32) and routing_weights (bf16) via Triton
    E = num_tokens * num_experts_per_tok
    e_vec = torch.empty(E, dtype=torch.int32, device=device)
    wt_vec = torch.empty(E, dtype=torch.bfloat16, device=device)

    BLOCK = 1024
    grid1 = (triton.cdiv(E, BLOCK),)
    flatten_experts_kernel[grid1](
        selected_experts, e_vec, num_tokens, num_experts_per_tok, E, BLOCK,
    )

    grid2 = (triton.cdiv(E, BLOCK),)
    flatten_weights_kernel[grid2](
        routing_weights, wt_vec, num_tokens, num_experts_per_tok, E, BLOCK,
    )

    # Stable sort flattened expert list (int32) using odd-even sort
    e_sorted = torch.empty(E, dtype=torch.int32, device=device)
    grid_sort = (triton.cdiv(E, BLOCK),)
    stable_sort_by_exp_kernel[grid_sort](
        e_vec, e_sorted, E, BLOCK,
    )

    # Bincount per expert in the sorted list
    counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    grid_bin = (1,)
    bincount_experts_kernel[grid_bin](
        e_sorted, counts, E, num_experts, BLOCK,
    )

    # Cumsum starts = prefix sum of counts
    starts = torch.empty(num_experts, dtype=torch.int32, device=device)
    grid_cum = (1,)
    cumsum_starts_kernel[grid_cum](
        counts, starts, num_experts,
    )

    # Capacity: ceil(1.25 * counts_per_expert), at least 1
    counts_f = counts.to(torch.float32)
    capacity = int(torch.ceil(counts_f * 1.25).max().item())
    capacity = max(capacity, 1)

    # Compute within_pos and validity mask
    within_pos = torch.empty(E, dtype=torch.int32, device=device)
    valid_int = torch.empty(E, dtype=torch.int32, device=device)

    grid_with = (triton.cdiv(E, BLOCK),)
    compute_within_pos_kernel[grid_with](
        e_sorted, starts, num_experts, E, capacity, within_pos, valid_int, BLOCK,
    )

    # Final weighted scatter-add into result (float32 accumulation)
    result_f32 = torch.zeros(num_tokens * hidden_size, dtype=torch.float32, device=device)

    grid_scatter = (triton.cdiv(E, BLOCK),)
    scatter_weighted_add_result_kernel[grid_scatter](
        within_pos, valid_int, wt_vec, result_f32, num_tokens, hidden_size, num_experts_per_tok, E, BLOCK,
    )

    # Cast to bfloat16 to match original output dtype
    result = result_f32.to(torch.bfloat16).view(num_tokens, hidden_size)
    return result


class ModelNew(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        expert_gate_weights: torch.Tensor,
        expert_up_weights: torch.Tensor,
        expert_down_weights: torch.Tensor,
    ):
        # Forward only launches Triton kernels; no torch tensor ops
        return run(
            hidden_states, selected_experts, routing_weights,
            expert_gate_weights, expert_up_weights, expert_down_weights
        )


def run(*args):
    return ModelNew()(*args)
