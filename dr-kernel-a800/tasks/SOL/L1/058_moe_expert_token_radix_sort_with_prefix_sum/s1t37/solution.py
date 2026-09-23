import triton
import triton.language as tl


@triton.jit
def histogram_out_kernel(flat_ptr, counts_ptr, N: tl.constexpr, num_experts: tl.constexpr, BLOCK: tl.constexpr):
    # One program per expert id
    id = tl.program_id(axis=0)
    count = tl.zeros((), dtype=tl.int32)

    # Iterate over flat in chunks of BLOCK
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32
        # Count occurrences of 'id' in this chunk
        for j in range(BLOCK):
            if mask[j]:
                if vals[j] == id:
                    count += 1

    # Write count for this expert
    tl.store(counts_ptr + id, count)


@triton.jit
def compute_le_counts_inclusive(counts_ptr, le_counts_ptr, num_experts: tl.constexpr):
    # Single-program inclusive scan on counts -> le_counts
    running = 0
    for i in range(num_experts):
        ci = tl.load(counts_ptr + i)
        running += ci
        tl.store(le_counts_ptr + i, running)


@triton.jit
def compute_inv_idx_kernel(flat_ptr, inv_idx_ptr, N: tl.constexpr, num_experts: tl.constexpr):
    # Single-program loop over flat to compute inv_idx per id: increment for each new occurrence
    for i in range(N):
        v = tl.load(flat_ptr + i)
        if (v >= 0) & (v < num_experts):
            old = tl.load(inv_idx_ptr + v)
            tl.store(inv_idx_ptr + v, old + 1)


@triton.jit
def compute_out_pos_real(flat_ptr, out_ptr, le_counts_ptr, inv_idx_ptr, N: tl.constexpr, num_experts: tl.constexpr):
    # Each program computes the output position for a single input index i
    for i in range(N):
        v = tl.load(flat_ptr + i)
        if (v >= 0) & (v < num_experts):
            le = tl.load(le_counts_ptr + v)
            inv = tl.load(inv_idx_ptr + v)
            # Stable tie-breaking: subtract 1 if inv>0, else 0
            pos = le - (inv > 0)
            # Store i at position 'pos'
            tl.store(out_ptr + pos, i)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Expect topk_idx: int32 tensor of shape (B, S, EPT), on CUDA device
        flat = topk_idx.contiguous().view(-1)
        N = flat.numel()
        num_experts = 256  # fixed in the original code

        # Allocate counts, le_counts, inv_idx on device
        counts = torch.empty(num_experts, dtype=torch.int32, device=flat.device)
        le_counts = torch.empty(num_experts, dtype=torch.int32, device=flat.device)
        inv_idx = torch.empty(num_experts, dtype=torch.int32, device=flat.device)

        # Launch histogram kernel: one program per expert
        grid_hist = (num_experts,)
        histogram_out_kernel[grid_hist](flat, counts, N, num_experts, BLOCK=1024)

        # Compute le_counts via inclusive scan
        compute_le_counts_inclusive[(1,)](counts, le_counts, num_experts)

        # Compute inv_idx for stable tie-breaking
        compute_inv_idx_kernel[(1,)](flat, inv_idx, N, num_experts)

        # Compute out_pos (stable argsort permutation) with grid=(N,)
        out_pos = torch.empty(N, dtype=torch.int32, device=flat.device)
        compute_out_pos_real[(N,)](flat, out_pos, le_counts, inv_idx, N, num_experts)

        # Reshape to original shape: (B, S, EPT)
        sorted_token_indices = out_pos.view(*topk_idx.shape)

        # Compute expert_offsets: inclusive prefix sums of counts
        # offsets: length num_experts+1, offsets[0] = 0
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        running = 0
        for i in range(num_experts):
            running += counts[i]
            offsets[i + 1] = running

        # Return sorted_token_indices and expert_offsets
        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
