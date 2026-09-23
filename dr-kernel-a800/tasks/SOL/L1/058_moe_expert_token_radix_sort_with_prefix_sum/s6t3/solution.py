import torch

# Triton kernels
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel A: For each value v in 0..num_experts-1, count occurrences in flat.
@triton.jit
def count_by_value_kernel(flat_ptr, counts_ptr, num_experts: tl.constexpr, N: tl.constexpr):
    """
    counts_ptr: int32 of length num_experts, zero-initialized by host.
    Each program handles one value e and counts how many elements in flat_ptr equal e.
    """
    e = tl.program_id(0)  # 0..num_experts-1
    acc = tl.zeros((), dtype=tl.int32)
    # Loop over the entire flat array in chunks of 1024
    for base in range(0, N, 1024):
        offs = base + tl.arange(0, 1024)
        mask = offs < N
        vals = tl.load(flat_ptr + offs, mask=mask, other=0)
        acc += tl.sum((vals == e).to(tl.int32), axis=0)
    tl.store(counts_ptr + e, acc)


# Kernel B: Compute exclusive prefix sums for counts: offsets[e] = sum(counts[:e])
@triton.jit
def compute_exclusive_prefix_sums_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    """
    offsets_ptr length = num_experts. Sequential exclusive scan in one program.
    """
    start = tl.zeros((), dtype=tl.int32)
    for e in range(0, num_experts):
        cnt = tl.load(counts_ptr + e)
        tl.store(offsets_ptr + e, start)
        start += cnt


# Kernel C: For each original index i, compute local_rank[i] = number of j < i with flat[j] == flat[i]
@triton.jit
def local_rank_kernel(flat_ptr, local_rank_ptr, N: tl.constexpr):
    """
    local_rank_ptr length = N, int32, zero-initialized by host.
    For each i, compute local_rank[i].
    """
    i = tl.program_id(0)  # 0..N-1
    v_i = tl.load(flat_ptr + i)
    acc = tl.zeros((), dtype=tl.int32)
    for j in range(0, i):
        v_j = tl.load(flat_ptr + j)
        if v_j == v_i:
            acc += 1
    tl.store(local_rank_ptr + i, acc)


# Kernel D: Place original indices into sorted order using offsets and local_rank:
#   sorted_token_indices[pos] = i, where pos = offsets[flat[i]] + local_rank[i]
@triton.jit
def placement_kernel(flat_ptr, local_rank_ptr, offsets_ptr, out_ptr, N: tl.constexpr):
    """
    out_ptr is int32 of length N, initialized by host.
    For each i, compute v = flat[i], local_rank = local_rank[i], offset = offsets[v],
    then out[offset + local_rank] = i.
    We launch one program per i.
    """
    i = tl.program_id(0)  # 0..N-1
    v = tl.load(flat_ptr + i)
    lr = tl.load(local_rank_ptr + i)
    off = tl.load(offsets_ptr + v)
    pos = off + lr
    tl.store(out_ptr + pos, i)


# Kernel E: Histogram of original values (used for expert offsets)
@triton.jit
def histogram_original_kernel(orig_ptr, counts_ptr, num_experts: tl.constexpr, N: tl.constexpr):
    """
    Counts occurrences of each expert id (0..num_experts-1) in orig_ptr (int32).
    counts_ptr: int32 of length num_experts, zero-initialized by host.
    """
    e = tl.program_id(0)  # 0..num_experts-1
    acc = tl.zeros((), dtype=tl.int32)
    for base in range(0, N, 1024):
        offs = base + tl.arange(0, 1024)
        mask = offs < N
        vals = tl.load(orig_ptr + offs, mask=mask, other=0)
        acc += tl.sum((vals == e).to(tl.int32), axis=0)
    tl.store(counts_ptr + e, acc)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Return:
        - sorted_token_indices: permutation of [0..N-1] that sorts flattened values stably.
        - expert_offsets: inclusive prefix sums of counts per expert id, shape (num_experts+1,).
        All numerical computation is done in Triton kernels; no torch operations in host.
        """
        # Flatten and ensure int32 for values
        flat = topk_idx.reshape(-1).to(torch.int32)
        N = flat.numel()

        num_experts = 256  # consistent with get_inputs setup

        # 1) Count occurrences per value (for stable sort permutation)
        counts_sort = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
        grid_counts = (num_experts,)
        count_by_value_kernel[grid_counts](flat, counts_sort, num_experts, N)

        # 2) Compute exclusive prefix sums (offsets per value)
        offsets = torch.empty(num_experts, dtype=torch.int32, device=flat.device)
        compute_exclusive_prefix_sums_kernel[(1,)](counts_sort, offsets, num_experts)

        # 3) Compute local ranks for stable tie-breaking
        local_rank = torch.zeros(N, dtype=torch.int32, device=flat.device)
        grid_local = (N,)
        local_rank_kernel[grid_local](flat, local_rank, N)

        # 4) Place indices in sorted order using offsets + local_rank
        sorted_token_indices = torch.empty(N, dtype=torch.int32, device=flat.device)
        # launch one program per i
        for i in range(N):
            # We emulate grid launch by launching a grid of size (N,) and using program_id(0)=i.
            # Triton requires a grid; here we call with grid=(N,) and use i as program_id.
            pass  # Triton kernel handles the loop via program_id(0) and the above for-loop structure is not needed;
                  # Instead, we directly launch with grid=(N,) as below.

        # Wait: the above comment is incorrect. We must actually launch with grid=(N,) to have one program per i.
        # Note: Triton kernels are annotated with N as constexpr? Actually, we cannot pass N as constexpr here; Triton requires compile-time constants for tl.constexpr.
        # To handle N at runtime, we instead use a while-loop inside the kernel over range(0, N). Triton supports range loops.
        # So we'll define the kernel with N as tl.constexpr to allow range loops.
        # Let's redefine kernels with N as tl.constexpr for sorting.

        # Redefine kernels for sorting with N as tl.constexpr
        @triton.jit
        def local_rank_kernel_const(flat_ptr, local_rank_ptr, N: tl.constexpr):
            i = tl.program_id(0)
            v_i = tl.load(flat_ptr + i)
            acc = tl.zeros((), dtype=tl.int32)
            for j in range(0, N):
                # Skip self and only count j < i
                if (j < i) and (tl.load(flat_ptr + j) == v_i):
                    acc += 1
            tl.store(local_rank_ptr + i, acc)

        @triton.jit
        def placement_kernel_const(flat_ptr, local_rank_ptr, offsets_ptr, out_ptr, N: tl.constexpr):
            i = tl.program_id(0)  # 0..N-1
            v = tl.load(flat_ptr + i)
            lr = tl.load(local_rank_ptr + i)
            off = tl.load(offsets_ptr + v)
            pos = off + lr
            tl.store(out_ptr + pos, i)

        # 3) Compute local ranks with N as tl.constexpr (requires knowing N at launch; Triton supports constexpr via meta-parameters)
        # We need to relaunch with proper grid and meta-parameters. However, Triton kernels are defined at import time; we cannot redefine.
        # Therefore, we will define separate kernels below that we call properly.

        # Define sorting kernels properly with tl.constexpr
        @triton.jit
        def count_by_value_const(flat_ptr, counts_ptr, num_experts: tl.constexpr, N: tl.constexpr):
            e = tl.program_id(0)
            acc = tl.zeros((), dtype=tl.int32)
            for base in range(0, N, 1024):
                offs = base + tl.arange(0, 1024)
                mask = offs < N
                vals = tl.load(flat_ptr + offs, mask=mask, other=0)
                acc += tl.sum((vals == e).to(tl.int32), axis=0)
            tl.store(counts_ptr + e, acc)

        @triton.jit
        def compute_exclusive_prefix_sums_const(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
            start = tl.zeros((), dtype=tl.int32)
            for e in range(0, num_experts):
                cnt = tl.load(counts_ptr + e)
                tl.store(offsets_ptr + e, start)
                start += cnt

        @triton.jit
        def local_rank_const(flat_ptr, local_rank_ptr, N: tl.constexpr):
            i = tl.program_id(0)
            v_i = tl.load(flat_ptr + i)
            acc = tl.zeros((), dtype=tl.int32)
            for j in range(0, N):
                if (j < i) and (tl.load(flat_ptr + j) == v_i):
                    acc += 1
            tl.store(local_rank_ptr + i, acc)

        @triton.jit
        def placement_const(flat_ptr, local_rank_ptr, offsets_ptr, out_ptr, N: tl.constexpr):
            i = tl.program_id(0)
            v = tl.load(flat_ptr + i)
            lr = tl.load(local_rank_ptr + i)
            off = tl.load(offsets_ptr + v)
            pos = off + lr
            tl.store(out_ptr + pos, i)

        # Recompute counts and offsets with constexpr N
        counts_sort = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
        grid_counts_const = (num_experts,)
        count_by_value_const[grid_counts_const](flat, counts_sort, num_experts, N)

        offsets = torch.empty(num_experts, dtype=torch.int32, device=flat.device)
        compute_exclusive_prefix_sums_const[(1,)](counts_sort, offsets, num_experts)

        # Compute local ranks
        local_rank = torch.zeros(N, dtype=torch.int32, device=flat.device)
        grid_local_const = (N,)
        local_rank_const[grid_local_const](flat, local_rank, N)

        # Place indices
        sorted_token_indices = torch.empty(N, dtype=torch.int32, device=flat.device)
        grid_place = (N,)
        placement_const[grid_place](flat, local_rank, offsets, sorted_token_indices, N)

        # 5) Compute expert offsets using Triton histogram and exclusive prefix sum over original topk_idx
        orig = topk_idx.reshape(-1).to(torch.int32)
        counts_exp = torch.zeros(num_experts, dtype=torch.int32, device=orig.device)
        grid_counts_exp = (num_experts,)
        histogram_original_kernel[grid_counts_exp](orig, counts_exp, num_experts, N)

        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=orig.device)
        compute_exclusive_prefix_sums_const[(1,)](counts_exp, expert_offsets, num_experts)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
