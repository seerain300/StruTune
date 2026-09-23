import torch
import triton
import triton.language as tl


@triton.jit
def init_topk_idx_rand_3d(topk_ptr, B, S, K, E, BLOCK: tl.constexpr):
    """
    Fill a 3D tensor topk_ptr of shape (B, S, K) with random int32 expert IDs in [0, E).
    We linearize the indices: for each (b, s, k) where b in [0, B), s in [0, S), k in [0, K),
    compute flat = b*S*K + s*K + k and write a random id.
    """
    pid = tl.program_id(0)
    total = B * S * K
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < total
    # Compute (b, s, k) for each offs:
    # offs = b*S*K + s*K + k
    K_const = K
    S_const = S
    b = offs // (S_const * K_const)
    rem = offs % (S_const * K_const)
    s = rem // K_const
    k = rem % K_const
    flat = b * (S_const * K_const) + s * K_const + k
    # Generate random id in [0, E). Triton provides tl.rand for random numbers.
    rnd = tl.rand()
    id = tl.cast(rnd * E, tl.int32)
    # Store id (int32) to topk_ptr[flat]
    tl.store(topk_ptr + flat, id, mask=mask)


@triton.jit
def histogram_experts(x_ptr, counts_ptr, N, E, BLOCK: tl.constexpr):
    """
    Compute counts per expert id in x_ptr of length N.
    Each program processes BLOCK elements; for each id in [0, E), it atomically adds
    the count of occurrences in its chunk to counts_ptr[id].
    """
    pid = tl.program_id(0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < N
    x_vals = tl.load(x_ptr + offs, mask=mask, other=0)
    for id in range(0, E):
        eq = (x_vals == id) & mask
        cnt = tl.sum(eq.to(tl.int32), axis=0)
        tl.atomic_add(counts_ptr + id, cnt)


@triton.jit
def inclusive_scan_counts(counts_ptr, offsets_ptr, E):
    """
    Compute inclusive prefix sums of counts_ptr into offsets_ptr[1..E],
    with offsets_ptr[0] initialized to 0 on host.
    """
    running = tl.zeros((), dtype=tl.int32)
    for i in range(0, E):
        running += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + (i + 1), running)


@triton.jit
def stable_counting_sort(x_ptr, starts_ptr, out_ptr, N, E):
    """
    Stable counting sort over x_ptr of length N by int32 ids in [0, E).
    out_ptr receives permutation indices. starts_ptr is length E, initialized to exclusive prefix sums.
    For each pos in [0, N), read id = x[pos], write out[starts[id]] = pos, then starts[id] += 1.
    This preserves original order for ties due to iterating pos in ascending order.
    """
    pos = 0
    while pos < N:
        id = tl.load(x_ptr + pos)
        start = tl.load(starts_ptr + id)
        tl.store(out_ptr + start, pos)
        tl.store(starts_ptr + id, start + 1)
        pos += 1


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int, batch_size: int, seq_len: int, num_experts_per_tok: int, device: torch.device):
        super().__init__()
        self.num_experts = num_experts
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_experts_per_tok = num_experts_per_tok
        self.device = device

    def forward(self):
        """
        Triton-only forward:
        - Generate topk_idx (B, S, K) with random expert IDs using Triton.
        - Flatten to 1D x.
        - Compute histogram counts, offsets (inclusive prefix), and stable permutation via Triton.
        Returns sorted_token_indices (int32, length B*S*K) and expert_offsets (int32, length num_experts+1).
        """
        B = self.batch_size
        S = self.seq_len
        K = self.num_experts_per_tok
        E = self.num_experts

        # Allocate and initialize topk_idx with random expert IDs via Triton
        topk_idx = torch.empty((B, S, K), dtype=torch.int32, device=self.device)
        total_tokens = B * S * K
        BLOCK_INIT = 4096  # large block to reduce kernel launches
        grid_init = (triton.cdiv(total_tokens, BLOCK_INIT),)
        init_topk_idx_rand_3d[grid_init](topk_idx, B, S, K, E, BLOCK=BLOCK_INIT)

        # Flatten for computation
        x = topk_idx.reshape(-1)

        # 1) Histogram of expert IDs using Triton
        counts = torch.zeros(E, dtype=torch.int32, device=self.device)
        BLOCK_HIST = 2048
        grid_hist = (triton.cdiv(total_tokens, BLOCK_HIST),)
        histogram_experts[grid_hist](x, counts, total_tokens, E, BLOCK=BLOCK_HIST)

        # 2) Prefix sum to produce expert_offsets using Triton (offsets[0] = 0 already in counts)
        offsets = torch.empty(E + 1, dtype=torch.int32, device=self.device)
        offsets.fill_(0)
        grid_scan = (1,)
        inclusive_scan_counts[grid_scan](counts, offsets, E)

        # 3) Stable counting sort using Triton to produce sorted_token_indices (permutation)
        out = torch.empty(total_tokens, dtype=torch.int32, device=self.device)
        starts = offsets.clone()  # exclusive prefix sums; offsets[0] == 0
        grid_sort = (1,)
        stable_counting_sort[grid_sort](x, starts, out, total_tokens, E)

        # Return sorted_token_indices (int32 permutation) and expert_offsets
        return out, offsets


def run(*args):
    return ModelNew()(*args)
