import torch
import triton
import triton.language as tl


@triton.jit
def histogram_experts(x_ptr, counts_ptr, N, E, BLOCK: tl.constexpr):
    # Simple per-block histogram: process BLOCK elements per program and write to counts using atomic_add
    pid = tl.program_id(0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < N
    x_vals = tl.load(x_ptr + offs, mask=mask, other=0)
    # Atomic add counts per id
    for id in range(0, E):
        eq = (x_vals == id) & mask
        cnt = tl.sum(eq.to(tl.int32), axis=0)
        tl.atomic_add(counts_ptr + id, cnt)


@triton.jit
def inclusive_scan_counts(counts_ptr, offsets_ptr, E):
    # Compute offsets[1..E] as inclusive prefix sums of counts; offsets[0] is set by host to 0
    running = tl.zeros((), dtype=tl.int32)
    for i in range(0, E):
        running += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + i + 1, running)


@triton.jit
def stable_counting_sort(x_ptr, starts_ptr, out_ptr, N, E):
    # Stable counting sort: for each position pos, write out[starts[x[pos]]] = pos; increment starts[x[pos]]
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

    def forward(self):
        # Generate inputs exactly as in the original get_inputs
        # Note: get_inputs is not available here; we replicate its behavior.
        # We assume num_experts is fixed at 256 as in the original code.
        total_tokens = self.batch_size * self.seq_len * self.num_experts_per_tok
        topk_idx = torch.randint(
            0, self.num_experts,
            (self.batch_size, self.seq_len, self.num_experts_per_tok),
            dtype=torch.int32,
            device=self.device
        )

        # Flatten to 1D for processing
        x = topk_idx.reshape(-1)  # int32 on device

        N = x.numel()
        E = self.num_experts

        # 1) Histogram of expert IDs using Triton
        counts = torch.empty(E, dtype=torch.int32, device=self.device)
        BLOCK = 1024
        grid_hist = (triton.cdiv(N, BLOCK),)
        histogram_experts[grid_hist](x, counts, N, E, BLOCK=BLOCK)

        # 2) Compute expert offsets via inclusive prefix sum in Triton
        offsets = torch.empty(E + 1, dtype=torch.int32, device=self.device)
        offsets.fill_(0)  # offsets[0] = 0
        grid_scan = (E,)
        inclusive_scan_counts[grid_scan](counts, offsets, E)

        # 3) Stable counting sort using Triton to produce sorted_token_indices (permutation)
        out = torch.empty(N, dtype=torch.int32, device=self.device)
        starts = offsets.clone()  # exclusive prefix sums; starts[e] = offsets[e]
        grid_sort = (1,)
        stable_counting_sort[grid_sort](x, starts, out, N, E)

        # Return sorted_token_indices (int32 permutation) and expert_offsets
        return out, offsets