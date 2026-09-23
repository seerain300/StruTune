import torch
import triton
import triton.language as tl


@triton.jit
def _pad_pos_kernel(out_ptr, N, B: tl.constexpr):
    """
    Pad positions [0..N-1] to length B (next power of two), out[i] = i for i < N, else 0.
    out_ptr points to an int32 tensor of length B.
    """
    pid = tl.program_id(0)
    if pid < B:
        i = pid
        if i < N:
            tl.store(out_ptr + i, i)
        else:
            tl.store(out_ptr + i, 0)


@triton.jit
def _bitonic_argsort_stable(out_ptr, pos_ptr, a_ptr, N, B: tl.constexpr):
    """
    Stable bitonic argsort over padded positions. We carry 'pos' (original indices) and
    compare them against values in 'a'. For stable tie-breaking, when values are equal,
    the lower original index precedes. 'out' is the final argsort positions (length N).
    We only write to positions < N; padded positions are ignored.
    """
    # We implement the classic bitonic network over B elements, updating 'pos' accordingly.
    # For each pair (i, i^j), determine which one should move to position p using stable tie-breaking.
    # We'll iterate through fixed stages; for simplicity and Triton compatibility, we use nested
    # loops over j and k. Triton can compile loops where bounds are constexpr.
    # Note: This kernel is complex; ensure B is next power of two >= N and is a constexpr.

    # Initialize: load original positions
    for i in range(B):
        if i < N:
            pos_i = tl.load(pos_ptr + i)

    # Bitonic sorting network: classic schedule
    # Outer stages
    for stage in range(1, B + 1):
        k = 1 << stage
        for j in range(stage - 1, -1, -1):
            stride = 1 << j
            p = i ^ stride  # position partner
            # Load current pos and partner pos (we only use i, p < N to write)
            # We need to decide whether i or p moves to position i in this stage.
            # Classic bitonic: if (i < p):
            #   if ((i & k) == 0): descending
            #   else: ascending
            # Note: We will not perform actual swapping here; instead, we compute which original
            # index should be placed at position i after sorting. Triton doesn't provide easy
            # in-place swap across arrays; instead we track pos[i] using the decision logic.
            # For simplicity and correctness, we rely on standard bitonic network; the decision
            # to update pos[i] is made by each process at position i based on comparisons to its partner.
            # Implementing that requires reading partner's pos and a; Triton allows reading via tl.load
            # with pointer arithmetic. We pass pos_ptr and a_ptr and compute decisions per i.
            # However, Triton doesn't support dynamic looping indices efficiently here; therefore,
            # we implement comparator per stage in a vectorized manner by having each thread represent
            # a position i, and decide the update for that i. We'll do this by recomputing partner and
            # loading partner pos and a for each i.

            # Compute partner
            partner = i ^ stride
            # Load values and positions for i and partner (both must be < N)
            valid_i = i < N
            valid_p = partner < N
            val_i = tl.load(a_ptr + i) if valid_i else float('inf')
            pos_i_loaded = tl.load(pos_ptr + i) if valid_i else 0
            val_p = tl.load(a_ptr + partner) if valid_p else float('inf')
            pos_p_loaded = tl.load(pos_ptr + partner) if valid_p else 0

            # Determine sort direction for this stage: ascending if (i & k) == 0 else descending
            asc = (i & k) == 0

            # Stable comparison: if values differ, use value; else use original index
            # Define 'better': whether i should precede p in ascending order (stable)
            if asc:
                better = (val_i < val_p) or ((val_i == val_p) and (pos_i_loaded < pos_p_loaded))
            else:
                better = (val_i > val_p) or ((val_i == val_p) and (pos_i_loaded > pos_p_loaded))

            # If i should be at position i, we keep pos[i]; if p should be at i, we set pos[i] = pos_p
            new_pos_i = pos_i_loaded if better else pos_p_loaded

            # Store updated position only if i < N
            if valid_i:
                tl.store(out_ptr + i, new_pos_i)


@triton.jit
def _histogram_kernel(in_ptr, out_ptr, N, num_buckets: tl.constexpr):
    """
    Compute histogram of int32 values in in_ptr of length N, into out_ptr[num_buckets].
    Uses one atomic add per element.
    """
    pid = tl.program_id(0)
    # Single program iterating over N; Triton supports loops where N is runtime, but here we
    # set grid=(1,) and loop over N. This is acceptable for moderate N.
    for i in range(0, N):
        val = tl.load(in_ptr + i)
        # Ensure val is int32
        val = tl.cast(val, tl.int32)
        if val >= 0 and val < num_buckets:
            tl.atomic_add(out_ptr + val, 1)


@triton.jit
def _inclusive_scan_prefix_sum(in_ptr, out_ptr, num_buckets: tl.constexpr):
    """
    Inclusive prefix sum over a small array (num_buckets=256) using iterative doubling.
    in_ptr: input histogram of length num_buckets
    out_ptr: output offsets of length num_buckets+1; out_ptr[0]=0, out_ptr[1..]=cumsum
    """
    # out_ptr[0] is set by host to 0
    # Compute scan in-place into out_ptr[1..]
    # Iterative doubling
    step = 1
    while step < num_buckets:
        # For each i, add out[i - step] if i >= step
        for i in range(1, num_buckets + 1):
            prev = i - step
            if prev >= 1:
                tl.atomic_add(out_ptr + i, tl.load(out_ptr + prev))
        step = step << 1


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation of:
          - sorted_token_indices: permutation of [0..N-1] that sorts flattened topk_idx (stable)
          - expert_offsets: cumulative counts per expert ID (num_experts=256), length 257
        """
        # Ensure we have a CUDA tensor
        device = topk_idx.device
        a = topk_idx.reshape(-1)
        # We need int32 for Triton kernels
        a32 = a.to(torch.int32).contiguous()
        N = a32.numel()

        # 1) Compute padded length B as next power of two >= N
        #    For Triton kernels, we need B as constexpr; pass as int
        b = 1
        while b < N:
            b <<= 1
        B = b

        # 2) Launch padding kernel to prepare initial positions
        pos = torch.empty(B, dtype=torch.int32, device=device)
        _pad_pos_kernel[(1,)](pos, N, B=B)

        # 3) Bitonic argsort stable: compute argsort permutation (length N)
        #    We will write the final argsort indices into 'out_pos' at positions < N.
        out_pos = torch.empty(B, dtype=torch.int32, device=device)
        _bitonic_argsort_stable[(1,)](out_pos, pos, a32, N, B=B)
        # Extract argsort indices for the first N positions
        sorted_token_indices = out_pos[:N].to(torch.int64)  # original code uses int64 indices

        # 4) Histogram of expert IDs (int32)
        histogram = torch.zeros(256, dtype=torch.int32, device=device)
        _histogram_kernel[(1,)](a32, histogram, N, num_buckets=256)

        # 5) Inclusive prefix sum to get expert offsets (length 257)
        offsets = torch.empty(257, dtype=torch.int32, device=device)
        offsets[0] = 0
        _inclusive_scan_prefix_sum[(1,)](histogram, offsets, num_buckets=256)

        # Return sorted_token_indices (1D, length N, int64) and expert_offsets (1D, length 257, int32)
        # Note: original code uses int32 for offsets; we can return int64 too. Either is fine.
        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
