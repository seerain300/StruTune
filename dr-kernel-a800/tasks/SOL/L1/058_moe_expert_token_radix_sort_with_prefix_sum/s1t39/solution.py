import triton
import triton.language as tl

# Triton kernel: build histogram of values in 'flat' without atomics.
# One program processes one expert k, loops over the entire flat array and increments counts[k].
@triton.jit
def histogram_out_kernel(flat_ptr, counts_ptr, N: tl.constexpr, NUM_EXPERTS: tl.constexpr):
    k = tl.program_id(axis=0)  # k in [0, NUM_EXPERTS)
    # Loop over all elements in flat to count occurrences of k
    for i in range(N):
        # load flat[i] (int32)
        val = tl.load(flat_ptr + i)  # val is int32
        # If val == k, increment counts[k]
        if val == k:
            # counts_ptr is int32
            curr = tl.load(counts_ptr + k)
            curr += 1
            tl.store(counts_ptr + k, curr)

# Triton kernel: compute inclusive prefix sums of 'input' into 'output' (single-program).
@triton.jit
def inclusive_scan_single_kernel(input_ptr, output_ptr, length: tl.constexpr):
    total = 0
    for i in range(length):
        total += tl.load(input_ptr + i)
        tl.store(output_ptr + i, total)

# Triton kernel: compute inv_idx per expert id = number of earlier elements with the same id.
# Single program loops over flat, increments inv_idx[id] each time id appears.
@triton.jit
def compute_inv_idx_kernel(flat_ptr, inv_idx_ptr, N: tl.constexpr, NUM_EXPERTS: tl.constexpr):
    for i in range(N):
        val = tl.load(flat_ptr + i)
        # val in [0, NUM_EXPERTS)
        # increment inv_idx[val]
        curr = tl.load(inv_idx_ptr + val)
        curr += 1
        tl.store(inv_idx_ptr + val, curr)

# Triton kernel: compute stable argsort permutation 'out_pos' for each element using le_counts and inv_idx.
# Single program loops over flat; for each i, it computes position pos based on id = flat[i].
@triton.jit
def compute_out_pos_kernel(flat_ptr, out_pos_ptr, le_counts_ptr, inv_idx_ptr, N: tl.constexpr, NUM_EXPERTS: tl.constexpr):
    for i in range(N):
        val = tl.load(flat_ptr + i)  # id
        le = tl.load(le_counts_ptr + val)  # inclusive count up to val
        inv = tl.load(inv_idx_ptr + val)   # number of earlier elements with same id
        pos = le - (inv if inv > 0 else 0)  # stable tie-breaking
        tl.store(out_pos_ptr + i, pos)

# Triton kernel: compute expert offsets as inclusive prefix sums of counts, writing into offsets[1:].
# offsets[0] is kept at 0 by the host.
@triton.jit
def compute_offsets_kernel(counts_ptr, offsets_ptr, NUM_EXPERTS: tl.constexpr):
    total = 0
    for k in range(NUM_EXPERTS):
        total += tl.load(counts_ptr + k)
        tl.store(offsets_ptr + 1 + k, total)

class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Expect topk_idx: int32 tensor of shape (B, S, EPT), on CUDA device.
        flat = topk_idx.view(-1).contiguous()
        N = flat.numel()
        num_experts = 256  # fixed as per original code

        # Allocate counts and le_counts as int32 on device
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
        le_counts = torch.empty(num_experts, dtype=torch.int32, device=flat.device)

        # 1) Build histogram counts per expert using Triton kernel (one program per expert)
        # Note: N is known at runtime; Triton supports loops over runtime integers.
        grid_hist = (num_experts,)
        histogram_out_kernel[grid_hist](flat, counts, N, num_experts)

        # 2) Compute le_counts (inclusive prefix sums) using single-program inclusive scan
        compute_le_counts_inclusive = inclusive_scan_single_kernel
        compute_le_counts_inclusive[(1,)](counts, le_counts, num_experts)

        # 3) Compute inv_idx per expert: number of earlier elements with the same id
        inv_idx = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
        compute_inv_idx_kernel[(1,)](flat, inv_idx, N, num_experts)

        # 4) Compute stable argsort permutation out_pos using le_counts and inv_idx
        out_pos = torch.empty(N, dtype=torch.int32, device=flat.device)
        compute_out_pos_kernel[(1,)](flat, out_pos, le_counts, inv_idx, N, num_experts)

        # 5) Compute expert_offsets in Triton: inclusive prefix sums of counts into offsets[1:]
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        # offsets[0] = 0 (left as zero by default); fill [1:]
        compute_offsets_kernel[(1,)](counts, offsets, num_experts)

        # Reshape out_pos back to original shape: (B, S, EPT)
        sorted_token_indices = out_pos.view(*topk_idx.shape)

        # Return the required outputs (sorted_token_indices and expert_offsets)
        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
