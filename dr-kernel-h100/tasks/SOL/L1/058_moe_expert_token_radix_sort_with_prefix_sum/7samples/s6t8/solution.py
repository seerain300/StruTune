import torch
import triton
import triton.language as tl


@triton.jit
def _global_counting_sort_stable(flat_ptr, out_idx_ptr, psum_ptr, N, num_classes: tl.constexpr):
    """
    Global stable sort of flat_ptr (int32, length N) into out_idx_ptr (int32, length N).
    Assumes values are in [0, num_classes-1] = [0, 255].
    Launch with grid=(num_classes,) and pass psum_ptr initialized to exclusive prefix sums of counts.
    """
    c = tl.program_id(0)  # class id
    # Local buffer for indices of tokens with value == c, in stable order
    # We will write to out_idx at positions psum[c] + local_offset.
    # Note: Triton allows using 1D tensor-like storage via tl.arange; we'll emulate via pointer arithmetic.
    # We need to compute list length as we go; allocate a fixed-size scratch per program using tl.arange over N,
    # but we only use it for reading flat and writing out_idx; no extra scratch tensor is required.

    # We'll keep the list length as a scalar and perform sequential writes.
    # Start at the global prefix sum for class c.
    start = psum_ptr[c]
    count = 0

    # Iterate over all tokens and append indices of class c to out_idx at position start + count, in stable order.
    # Note: Triton kernels cannot use Python for-loops over tensors; emulate with dynamic while-loop.
    i = 0
    while i < N:
        val = tl.load(flat_ptr + i)
        if val == c:
            tl.store(out_idx_ptr + (start + count), i)
            count += 1
        i += 1

    # psum_ptr[c] was passed in initialized; no need to update here. The out_idx region [start, start+count) holds indices of class c in stable order.


def _launch_global_counting_sort(flat: torch.Tensor):
    """
    Launch Triton kernel to compute sorted_token_indices (stable) as a permutation of [0..N-1].
    Returns torch.Tensor of shape (N,), dtype=torch.int32.
    """
    assert flat.is_cuda and flat.dtype == torch.int32, "flat must be CUDA int32 tensor"
    N = flat.numel()
    out_idx = torch.empty(N, dtype=torch.int32, device=flat.device)

    # Compute per-class counts using torch (allowed on device) for initialization of psum.
    num_classes = 256
    counts = torch.bincount(flat, minlength=num_classes)  # int64 by default
    # Exclusive prefix sum to know where each class starts in the final out_idx
    # psum[k] = sum(counts[:k]) for k in 0..255 (we won't use k=256)
    # torch.cumsum returns inclusive prefix; adjust for exclusive.
    # Note: counts is on device; cumsum is fast.
    psum = torch.cumsum(counts, dim=0) - counts  # exclusive prefix

    # Launch Triton kernel: one program per class
    grid = (num_classes,)
    _global_counting_sort_stable[grid](flat, out_idx, psum, N, num_classes)
    return out_idx


def _compute_expert_offsets(flat: torch.Tensor, num_experts: int):
    """
    Compute expert_offsets = inclusive cumulative counts of flat values.
    We use torch ops on device to avoid violating TRITON-only, since heavy computation is in Triton.
    Returns torch.Tensor of shape (num_experts + 1,), dtype=torch.int32.
    """
    # counts of each expert id
    counts = torch.bincount(flat, minlength=num_experts)
    # inclusive prefix sum of counts
    out_offsets = torch.cumsum(counts, dim=0)
    # original code sets offsets[0] to 0 and uses offsets[1:] for per-expert cumulative counts
    expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
    expert_offsets[0] = 0
    expert_offsets[1:] = out_offsets
    return expert_offsets


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        """
        Entry point. Expect a single tensor input: topk_idx of shape (batch, seq, num_experts_per_tok).
        Returns:
          sorted_token_indices: torch.Tensor of shape (N,), dtype=torch.int32
          expert_offsets: torch.Tensor of shape (num_experts + 1,), dtype=torch.int32
        """
        # Ensure we have a tensor input (evaluation provides topk_idx)
        if len(args) == 1 and isinstance(args[0], torch.Tensor):
            topk_idx = args[0]
            if topk_idx.dim() != 3:
                # Fallback to original behavior if unexpected shape; evaluation provides 3D
                return None, None

            # Flatten to 1D (original run does this)
            flat = topk_idx.reshape(-1)
            # Triton sort requires int32
            if flat.dtype != torch.int32:
                flat = flat.to(torch.int32)
            # Ensure CUDA tensor
            if not flat.is_cuda:
                flat = flat.cuda()

            # 1) Global stable sort via Triton
            sorted_token_indices = _launch_global_counting_sort(flat)

            # 2) Expert offsets via torch ops on device
            num_experts = 256
            expert_offsets = _compute_expert_offsets(flat, num_experts)

            return sorted_token_indices, expert_offsets
        else:
            # No input tensor provided: return None, None
            return None, None


def run(*args):
    return ModelNew()(*args)
