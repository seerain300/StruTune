import torch
import triton
import triton.language as tl


@triton.jit
def histogram_experts(x_ptr, counts_ptr, N, E, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < N
    x_vals = tl.load(x_ptr + offs, mask=mask, other=0)
    # For each expert id, count occurrences in this block and atomic add to global counts
    for id in range(0, E):
        eq = (x_vals == id) & mask
        cnt = tl.sum(eq.to(tl.int32), axis=0)
        tl.atomic_add(counts_ptr + id, cnt)


@triton.jit
def inclusive_scan_counts(counts_ptr, offsets_ptr, E):
    # Compute offsets[1..E] = inclusive prefix sum of counts (exclusive at 0)
    running = tl.zeros((), dtype=tl.int32)
    for i in range(0, E):
        running += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + i + 1, running)


@triton.jit
def stable_counting_sort(x_ptr, starts_ptr, out_ptr, N, E):
    # Fill out permutation stably:
    # For each position pos in [0, N), read expert id = x[pos], write out[start[e]] = pos, then start[e] += 1.
    pos = 0
    while pos < N:
        # Load x[pos]; since pos may exceed N in vectorized store, we scalar iterate pos
        # but Triton kernels support scalar control flow here.
        id = tl.load(x_ptr + pos)
        start = tl.load(starts_ptr + id)
        tl.store(out_ptr + start, pos)
        # Increment start for this expert
        # Note: starts_ptr points to int32 array of length E. We must ensure out_ptr is large enough.
        tl.store(starts_ptr + id, start + 1)
        pos += 1


class ModelNew(torch.nn.Module):
    def __init__(self, batch_size: int, seq_len: int, num_experts: int, num_experts_per_tok: int, device: torch.device):
        super().__init__()
        # Store axes to recreate x shape
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.device = device

    def forward(self):
        # Generate topk_idx as in original get_inputs
        topk_idx = torch.randint(0, self.num_experts,
                                  (self.batch_size, self.seq_len, self.num_experts_per_tok),
                                  dtype=torch.int32,
                                  device=self.device)
        x = topk_idx.reshape(-1)  # 1D flattened tensor

        N = x.numel()
        E = self.num_experts

        # 1) Histogram of expert IDs using Triton
        counts = torch.zeros(E, dtype=torch.int32, device=self.device)
        BLOCK = 1024  # chunk size for histogram kernel
        grid = (triton.cdiv(N, BLOCK),)
        histogram_experts[grid](x, counts, N, E, BLOCK=BLOCK)

        # 2) Compute expert offsets via inclusive prefix sum in Triton
        offsets = torch.empty(E + 1, dtype=torch.int32, device=self.device)
        offsets.fill_(0)  # offsets[0] will be 0, offsets[1..E] computed by kernel
        # Start positions per expert: we need starts[i] = offsets[i], so we can reuse offsets buffer
        # But kernel inclusive_scan_counts expects starts_ptr to receive cumulative sums.
        # We'll run the kernel to fill offsets[1..E] from counts.
        inclusive_scan_counts[(E,)](counts, offsets, E)

        # 3) Stable counting sort using Triton (fill permutation out)
        # out holds sorted_token_indices
        out = torch.empty(N, dtype=torch.int32, device=self.device)
        # starts per expert: initialize starts[e] = offsets[e] (exclusive prefix)
        starts = offsets.clone()
        # Now run stable sorting: for each pos, write out[starts[e]] = pos, then starts[e] += 1
        stable_counting_sort[(1,)](x, starts, out, N, E)

        # Return sorted_token_indices and expert_offsets
        return out, offsets


def run(*args):
    return ModelNew()(*args)
