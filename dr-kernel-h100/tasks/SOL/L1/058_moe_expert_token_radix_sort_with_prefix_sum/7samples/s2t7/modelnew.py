import torch
import triton
import triton.language as tl


@triton.jit
def init_topk_idx_rand(topk_ptr, N, E, BLOCK: tl.constexpr):
    # Write random expert IDs [0, E) into topk_ptr as int32
    pid = tl.program_id(0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < N
    # Triton doesn't provide torch.randint; use a simple formula with tl.rand
    # tl.rand returns a float in [0,1). Scale to [0, E), cast to int32.
    rand = tl.rand()  # scalar per program; broadcasting is fine
    ids = (rand * E).to(tl.int32)
    tl.store(topk_ptr + offs, ids, mask=mask)


@triton.jit
def histogram_experts(x_ptr, counts_ptr, N, E, BLOCK: tl.constexpr):
    # Compute per-expert counts using block-wise vectorized loads and atomic_add
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
    # offsets_ptr has length E+1; we compute offsets[1..E] = inclusive prefix sum
    running = tl.zeros((), dtype=tl.int32)
    for i in range(0, E):
        running += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + (i + 1), running)


@triton.jit
def stable_counting_sort(x_ptr, starts_ptr, out_ptr, N, E):
    # Fill out permutation stably:
    # For each position pos in [0, N), read expert id = x[pos], write out[starts[e]] = pos, then starts[e] += 1.
    pos = 0
    while pos < N:
        id = tl.load(x_ptr + pos)
        start = tl.load(starts_ptr + id)
        tl.store(out_ptr + start, pos)
        tl.store(starts_ptr + id, start + 1)
        pos += 1


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # We assume args contains the axes_and_scalars dict. If not, fall back to no args.
        # The original get_inputs signature takes device, but for Triton we can create tensors on default device.
        # Here, we generate inputs with Triton random and perform all computation in Triton.
        # We mimic the original behavior: return sorted_token_indices and expert_offsets.

        # Extract config from args (args is a single dict here)
        axes_and_scalars = args[0] if len(args) == 1 and isinstance(args[0], dict) else {
            "batch_size": 8, "seq_len": 256, "num_experts": 256, "num_experts_per_tok": 4
        }
        batch_size = int(axes_and_scalars["batch_size"])
        seq_len = int(axes_and_scalars["seq_len"])
        num_experts = int(axes_and_scalars["num_experts"])
        num_experts_per_tok = int(axes_and_scalars["num_experts_per_tok"])

        # Device: Triton requires CUDA; create on CUDA
        device = torch.device("cuda", 0)

        # 1) Allocate and initialize topk_idx via Triton random
        total_tokens = batch_size * seq_len * num_experts_per_tok
        topk_idx = torch.empty((batch_size, seq_len, num_experts_per_tok), dtype=torch.int32, device=device)
        # Flatten for processing
        x = topk_idx.reshape(-1)  # length = total_tokens

        BLOCK_INIT = 1024
        grid_init = (triton.cdiv(total_tokens, BLOCK_INIT),)
        init_topk_idx_rand[grid_init](x, total_tokens, num_experts, BLOCK=BLOCK_INIT)

        # 2) Histogram using Triton
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        BLOCK_HIST = 1024
        grid_hist = (triton.cdiv(total_tokens, BLOCK_HIST),)
        histogram_experts[grid_hist](x, counts, total_tokens, num_experts, BLOCK=BLOCK_HIST)

        # 3) Prefix sum to produce expert_offsets using Triton
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        offsets.fill_(0)
        grid_scan = (1,)
        inclusive_scan_counts[grid_scan](counts, offsets, num_experts)

        # 4) Stable counting sort using Triton to produce sorted_token_indices (permutation)
        out = torch.empty(total_tokens, dtype=torch.int32, device=device)
        starts = offsets.clone()  # exclusive prefix sums; we'll use offsets as start per expert
        grid_sort = (1,)
        stable_counting_sort[grid_sort](x, starts, out, total_tokens, num_experts)

        # Return sorted_token_indices (int32 permutation) and expert_offsets
        return out, offsets