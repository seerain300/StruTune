import torch
import triton
import triton.language as tl


@triton.jit
def histogram_experts(x_ptr, counts_ptr, N, E, BLOCK: tl.constexpr):
    # Compute counts per expert id in [0, E)
    for id in range(0, E):
        acc = tl.zeros((), dtype=tl.int32)
        start = 0
        while start < N:
            offs = start + tl.arange(0, BLOCK)
            mask = offs < N
            x_vals = tl.load(x_ptr + offs, mask=mask, other=0)
            eq = (x_vals == id) & mask
            acc += tl.sum(eq.to(tl.int32), axis=0)
            start += BLOCK
        tl.store(counts_ptr + id, acc)


@triton.jit
def inclusive_scan_counts(counts_ptr, offsets_ptr, E):
    # offsets[0] is already 0 on host; we compute offsets[1..E]
    running = tl.zeros((), dtype=tl.int32)
    i = 0
    while i < E:
        running += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + (i + 1), running)
        i += 1


@triton.jit
def stable_counting_sort(x_ptr, starts_ptr, out_ptr, N, E):
    # Fill out permutation stably:
    # For each position pos in [0, N), read id = x[pos], write out[starts[id]] = pos, then starts[id] += 1.
    pos = 0
    while pos < N:
        id = tl.load(x_ptr + pos)
        start = tl.load(starts_ptr + id)
        tl.store(out_ptr + start, pos)
        tl.store(starts_ptr + id, start + 1)
        pos += 1


class ModelNew(torch.nn.Module):
    def __init__(self, batch_size: int, seq_len: int, num_experts: int, num_experts_per_tok: int, device: torch.device):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.device = device

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation:
        - Flattens topk_idx (1D int32).
        - Computes histogram per expert with Triton.
        - Computes expert_offsets with Triton prefix sum.
        - Produces sorted_token_indices via stable Triton counting sort.
        """
        # topk_idx is expected to be provided by get_inputs; do not use any torch ops on it other than reshape
        # Flatten to 1D. torch.reshape is acceptable here because it is not a compute op, but forward should avoid other torch ops.
        x = topk_idx.reshape(-1)  # int32 on device

        E = self.num_experts
        N = x.numel()

        # 1) Histogram of expert IDs using Triton
        counts = torch.empty(E, dtype=torch.int32, device=self.device)
        histogram_experts[(1,)](x, counts, N, E, BLOCK=1024)

        # 2) Compute expert offsets via inclusive prefix sum in Triton
        offsets = torch.empty(E + 1, dtype=torch.int32, device=self.device)
        offsets.fill_(0)  # offsets[0] = 0
        inclusive_scan_counts[(E,)](counts, offsets, E)

        # 3) Stable counting sort using Triton to produce sorted_token_indices (permutation)
        out = torch.empty(N, dtype=torch.int32, device=self.device)
        starts = offsets.clone()  # exclusive prefix sums; starts[e] = offsets[e]
        stable_counting_sort[(1,)](x, starts, out, N, E)

        # Return sorted_token_indices (int32 permutation) and expert_offsets
        return out, offsets


def run(*args):
    return ModelNew()(*args)
