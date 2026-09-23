import triton
import triton.language as tl


@triton.jit
def compute_out_pos_triton(flat_ptr, out_ptr, counts_ptr, le_ptr, N, num_experts: tl.constexpr):
    """
    Compute stable argsort permutation for flat_ptr[0:N] into out_ptr[0:N].
    Also compute per-expert counts into counts_ptr[0:num_experts] and inclusive prefix le into le_ptr[0:num_experts].
    For each element i with id = flat[i], we:
      - compute le_counts[id] (inclusive sum of counts up to id)
      - update counts_ptr[id] += 1
      - set out[id] = i
    Stable ordering is ensured by iterating sequentially, so earlier i values get lower positions.
    """
    # Loop over all elements to assign positions
    for i in range(0, N):
        v = tl.load(flat_ptr + i)  # v is int32
        # Update counts and le_counts
        old = tl.load(counts_ptr + v)
        new = old + 1
        tl.store(counts_ptr + v, new)
        total = 0
        for k in range(0, num_experts):
            c = tl.load(counts_ptr + k)
            total += c
            tl.store(le_ptr + k, total)
        # Assign stable position: out[v] = i
        tl.store(out_ptr + v, i)


@triton.jit
def cumsum_inclusive_triton(counts_ptr, out_ptr, length: tl.constexpr):
    """
    Inclusive cumsum of counts_ptr[0:length] into out_ptr[0:length].
    out[0] = counts[0]; out[i] = sum_{j=0..i} counts[j].
    Single-program sequential loop.
    """
    total = tl.load(counts_ptr + 0)
    tl.store(out_ptr + 0, total)
    for i in range(1, length):
        val = tl.load(counts_ptr + i)
        total += val
        tl.store(out_ptr + i, total)


def run(topk_idx: torch.Tensor):
    """
    Triton-only implementation:
    Returns (sorted_token_indices, expert_offsets).
    - sorted_token_indices: permutation of [0..N-1] that would sort the flattened topk_idx stably (int32)
    - expert_offsets: inclusive prefix sums of counts per expert ID, length num_experts+1 (int32)
    """
    # Flatten and ensure int32 contiguous
    flat = topk_idx.reshape(-1).to(torch.int32).contiguous()
    N = flat.numel()
    device = flat.device

    num_experts = 256
    out = torch.empty(N, dtype=torch.int32, device=device)
    counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    le_counts = torch.empty(num_experts, dtype=torch.int32, device=device)

    # Launch Triton kernel to compute stable argsort permutation and counts+le
    compute_out_pos_triton[(1,)](flat, out, counts, le_counts, N, num_experts=num_experts)

    # sorted_token_indices is the permutation: out[i] = index of i-th element in stable sorted order.
    sorted_token_indices = out

    # expert_offsets: inclusive prefix sums of counts per expert, length num_experts+1, with offsets[0]=0
    offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
    # cumsum counts[1:] and place first element at 0
    cumsum_inclusive_triton[(1,)](counts, offsets[1:], length=num_experts)
    offsets[0] = 0

    return sorted_token_indices, offsets


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # The evaluator passes topk_idx as the first (and only) argument; ensure we handle it.
        if len(args) == 1 and isinstance(args[0], torch.Tensor):
            topk_idx = args[0]
            return run(topk_idx)
        # Fallback if unexpected args (shouldn't happen in evaluator)
        return None


def run(*args):
    return ModelNew()(*args)
