import triton
import triton.language as tl


# Kernel 1: Build counts per expert, one program per expert.
# Each program scans the entire flat array and increments counts[k] for each occurrence of k.
@triton.jit
def histogram_out_kernel(flat_ptr, counts_ptr, N, num_experts: tl.constexpr):
    k = tl.program_id(axis=0)  # expert id
    # Scan over the whole array; use a fixed loop bound N (constexpr expected at launch)
    for i in range(N):
        val = tl.load(flat_ptr + i)
        # If val == k, increment counts[k]
        if val == k:
            # counts_ptr is int32; increment by 1
            tl.store(counts_ptr + k, tl.load(counts_ptr + k) + 1)


# Kernel 2: Inclusive prefix sum of counts -> le_counts
@triton.jit
def compute_le_counts_inclusive(counts_ptr, le_counts_ptr, num_experts: tl.constexpr):
    acc = 0
    for j in range(num_experts):
        acc += tl.load(counts_ptr + j)
        tl.store(le_counts_ptr + j, acc)


# Kernel 3: Compute inv_idx[k] = number of earlier elements with id=k (stable tie-breaking).
@triton.jit
def compute_inv_idx(flat_ptr, inv_idx_ptr, N, num_experts: tl.constexpr):
    for i in range(N):
        val = tl.load(flat_ptr + i)
        tl.store(inv_idx_ptr + val, tl.load(inv_idx_ptr + val) + 1)


# Kernel 4: Compute stable argsort permutation. out_pos[i] = i; each program writes pos of i.
@triton.jit
def compute_out_pos_real(flat_ptr, out_pos_ptr, le_counts_ptr, inv_idx_ptr, N, num_experts: tl.constexpr):
    # This kernel runs with grid=(1,), i.e., single program; still, we keep loop form generic.
    for i in range(N):
        id = tl.load(flat_ptr + i)
        le = tl.load(le_counts_ptr + id)
        inv = tl.load(inv_idx_ptr + id)
        # Stable position: le - (inv if inv > 0 else 0)
        pos = le - tl.where(inv > 0, inv, 0)
        # out_pos is 1D; out_pos[pos] = i
        tl.store(out_pos_ptr + pos, i)


# Kernel 5: Compute inclusive prefix sum of counts into offsets[1:], with offsets[0] = 0.
@triton.jit
def compute_expert_offsets_inclusive(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    acc = 0
    # offsets_ptr is 1-indexed: offsets[0] should be handled on host (set to 0)
    for j in range(num_experts):
        acc += tl.load(counts_ptr + j)
        # write to offsets[1 + j]
        tl.store(offsets_ptr + 1 + j, acc)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure we are on CUDA and dtype is int32
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels."
        assert topk_idx.dtype == torch.int32, "topk_idx must be int32."

        # Flatten and ensure contiguous
        flat = topk_idx.view(-1).contiguous()
        N = flat.numel()
        num_experts = 256  # as per original code

        # Allocate buffers
        counts = torch.empty(num_experts, dtype=torch.int32, device=flat.device)
        le_counts = torch.empty(num_experts, dtype=torch.int32, device=flat.device)
        inv_idx = torch.empty(num_experts, dtype=torch.int32, device=flat.device)

        # 1) Build histogram: one program per expert
        grid_hist = (num_experts,)
        histogram_out_kernel[grid_hist](flat, counts, N, num_experts)

        # 2) Inclusive prefix sum of counts -> le_counts
        compute_le_counts_inclusive[(1,)](counts, le_counts, num_experts)

        # 3) Compute inv_idx (stable tie-breaking)
        compute_inv_idx[(1,)](flat, inv_idx, N, num_experts)

        # 4) Compute stable argsort permutation (out_pos[i] = i, out_pos[pos] = i)
        out_pos = torch.empty(N, dtype=torch.int32, device=flat.device)
        compute_out_pos_real[(1,)](flat, out_pos, le_counts, inv_idx, N, num_experts)

        # Reshape back to original shape for sorted_token_indices
        sorted_token_indices = out_pos.view(*topk_idx.shape)

        # 5) Compute expert_offsets via inclusive prefix sum of counts
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        # offsets[0] = 0
        offsets[0] = 0
        compute_expert_offsets_inclusive[(1,)](counts, offsets, num_experts)

        # Return the outputs: sorted_token_indices and expert_offsets
        # The original signature returns (sorted_token_indices, expert_offsets).
        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
