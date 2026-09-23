import triton
import triton.language as tl


@triton.jit
def histogram_atomic_kernel(flat_ptr, counts_ptr, N, BLOCK_SIZE: tl.constexpr, MAX_VAL: tl.constexpr):
    """
    Parallel histogram using atomic_add.
    Each program handles BLOCK_SIZE elements; masked loads; atomically increments counts[flat[i]].
    Assumes flat_ptr values in [0, MAX_VAL] (MAX_VAL=256 here).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)
    vals = vals.to(tl.int32)
    # Only attempt atomic_add for in-range values; otherwise, mask out.
    in_range = (vals >= 0) & (vals < MAX_VAL) & mask
    tl.atomic_add(counts_ptr + vals, 1, mask=in_range)


@triton.jit
def compute_inclusive_prefix(le_counts_ptr, counts_ptr, M: tl.constexpr):
    """
    Single-program inclusive prefix sum for le_counts:
    For j in [0..M-1]: le_counts[j] = sum_{i<=j} counts[i]
    """
    for j in tl.static_range(0, M):
        acc = 0
        # sum counts up to j
        for i in tl.static_range(0, j + 1):
            ci = tl.load(counts_ptr + i)
            acc += ci
        tl.store(le_counts_ptr + j, acc)


@triton.jit
def compute_out_pos_real(flat_ptr, sorted_ptr, taken_ptr, le_counts_ptr, N, M: tl.constexpr):
    """
    Compute stable argsort permutation:
    - For each expert k in 0..M-1:
      - cnt = counts[k] (read from global memory)
      - base_pos = le_counts[k] - cnt
      - First pass: mark duplicates in taken (for those i with flat[i] == k and already taken[i] == 0, set taken[i]=1)
      - Second pass: write positions: for i with flat[i] == k and taken[i] == 0, place at base_pos += 1, mark taken[i]=1
    Writes sorted_ptr[i] = position of i in stable order.
    """
    # Loop over k; Triton allows loops; M is constexpr => unrolled.
    for k in tl.static_range(0, M):
        # Read counts[k] and le_counts[k]
        cnt_k = tl.load(counts_ptr + k)
        le_k = tl.load(le_counts_ptr + k)
        # base position for this k, minus duplicates adjustment
        base = le_k - cnt_k

        # First pass: count duplicates and mark them in taken
        # We need to know how many elements with flat==k are not taken. We can't write directly,
        # but we can count via scanning; simpler: after computing cnt_k, mark duplicates by scanning
        # but since we only know duplicates after computing global pos, we instead rely on second pass.
        # However, to implement duplicate handling, we need total elements with flat==k and whether they are already placed.
        # The practical way here is to iterate over all i positions and update taken accordingly.
        # We'll do two phases: first, detect duplicates and mark them; then write positions.

        # We cannot branch per element directly here; instead, we rely on the second pass which re-reads flat
        # and takes into account whether it was marked in the first pass. To make this explicit, we emulate:
        # We do not have an intermediate global duplicate count; instead, we handle duplicates implicitly
        # by the condition in the second pass: we only place those i that are not marked in taken (i.e., not duplicates).
        # However, duplicates are those i where another j has the same k and we've already placed j. Since we cannot
        # access 'placed' per element, we instead mark duplicates in the first pass: any i with flat[i] == k
        # and already taken[i] == 1 is a duplicate. We scan i from 0 to N-1, but in Triton we need static loops.
        # Therefore, we simplify: duplicates handling is implicit via taken flag; in second pass, only those
        # with taken==0 get placed and take the base_pos, otherwise we skip. This preserves stability:
        # earlier indices not placed first take lower positions.

        # Second pass: write positions
        for i in tl.static_range(0, N):
            # Load flat[i]
            val_i = tl.load(flat_ptr + i).to(tl.int32)
            is_k = val_i == k
            not_taken = tl.load(taken_ptr + i) == 0
            should_place = is_k & not_taken
            # If placing, store base_pos and increment base; else do nothing
            # Note: Triton doesn't support dynamic indexing on tensors for store; we use scalar store via pointers.
            # Compute pointer for sorted_ptr[i]
            if should_place:
                tl.store(sorted_ptr + i, base)
                base += 1
                # mark taken[i] = 1
                tl.store(taken_ptr + i, 1)
            # no else needed; we only place one per original index among duplicates.


@triton.jit
def compute_expert_offsets_histogram(flat_ptr, offsets_ptr, N, BLOCK_SIZE: tl.constexpr, MAX_VAL: tl.constexpr):
    """
    Compute expert offsets via histogram and inclusive prefix sums, writing into offsets_ptr[1..]
    offsets_ptr[0] should be set to 0 on host side before launch.
    """
    # We'll run histogram_atomic_kernel to get counts
    # Then compute inclusive prefix sum for counts to produce le_counts (length MAX_VAL+1, but we'll store only [1..])
    # However, Triton cannot return arrays; we do the prefix sum in a single program.
    pass  # Placeholder; see forward for actual implementation.


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 256
        # For histogram, choose a reasonable block size; will be passed as constexpr.
        self.block_hist = 1024

    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D int32 on device
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        device = flat.device
        M = self.num_experts

        # Allocate outputs
        # sorted_token_indices: length N, int32
        sorted_indices = torch.empty(N, dtype=torch.int32, device=device)
        # expert_offsets: length M+1, int32, offsets[0]=0, [1..]=inclusive counts
        expert_offsets = torch.empty(M + 1, dtype=torch.int32, device=device)
        expert_offsets[0] = 0

        # 1) Histogram of flat values
        counts = torch.zeros(M, dtype=torch.int32, device=device)
        grid_hist = (triton.cdiv(N, self.block_hist),)
        histogram_atomic_kernel[grid_hist](flat, counts, N, BLOCK_SIZE=self.block_hist, MAX_VAL=M)

        # 2) Inclusive prefix sums le_counts for offsets
        le_counts = torch.empty(M, dtype=torch.int32, device=device)
        compute_inclusive_prefix[le_counts](le_counts, counts, M)

        # Store expert offsets [1..] (exclude leading zero for now; will fix later)
        expert_offsets[1:] = le_counts

        # 3) Compute stable argsort permutation via Triton kernel
        taken = torch.zeros(N, dtype=torch.int32, device=device)  # 0 means not taken, 1 means taken/duplicate for k
        # We need to compute le_counts for k too; we already have counts -> le_counts via compute_inclusive_prefix above.
        le_counts_k = le_counts  # length M
        # Launch compute_out_pos_real; it will fill sorted_indices
        compute_out_pos_real[(1,)](flat, sorted_indices, taken, le_counts_k, N, M=M)  # single-program launch; unrolled loops

        # Return results as required by original: (sorted_token_indices, expert_offsets)
        # Note: The original returns sorted_token_indices as int32 and expert_offsets with leading zero already set.
        return sorted_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
