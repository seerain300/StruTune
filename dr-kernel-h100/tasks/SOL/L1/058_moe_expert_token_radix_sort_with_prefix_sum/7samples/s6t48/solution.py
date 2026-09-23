import torch
import triton
import triton.language as tl


@triton.jit
def _hist_and_offsets_kernel(flat_ptr, counts_ptr, scan_ptr, N, NUM_CLASSES: tl.constexpr):
    # flat_ptr: *int32, length N
    # counts_ptr: *int32, length NUM_CLASSES (256)
    # scan_ptr: *int32, length (NUM_CLASSES + 1) (257), will hold inclusive scan of counts
    # We do histogram via atomics
    for i in range(NUM_CLASSES):
        # count how many elements in flat are equal to i
        cnt = 0
        for j in range(0, N):
            val = tl.load(flat_ptr + j)
            if val == i:
                cnt += 1
        # atomic add to global counts[i]
        tl.atomic_add(counts_ptr + i, cnt)

    # Initialize scan[0] = 0, scan[1:] = 0
    tl.store(scan_ptr + 0, 0)
    for i in range(1, NUM_CLASSES + 1):
        tl.store(scan_ptr + i, 0)

    # Inclusive scan of counts: scan[c+1] = scan[c] + counts[c]
    running = 0
    for c in range(NUM_CLASSES):
        cnt = tl.load(counts_ptr + c)
        running += cnt
        tl.store(scan_ptr + c + 1, running)

    # Compute and store sorted_token_indices via stable counting sort
    out_idx = tl.zeros([N], dtype=tl.int64)  # we will write scalar stores per i
    # Fill out_idx by class in stable order
    for c in range(NUM_CLASSES):
        start = tl.load(scan_ptr + c)         # int32
        end = tl.load(scan_ptr + c + 1)       # int32
        # Iterate over all positions
        for j in range(0, N):
            val = tl.load(flat_ptr + j)
            if val == c:
                tl.store(out_idx + start, tl.cast(j, tl.int64))
                start += 1
                # Optionally pad if needed (scan ensures we fill up to end, but we still iterate)
        # If start < end, we would need to pad zeros, but scan guarantees all positions are covered.


def _launch_hist_and_offsets(topk_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Ensure 3D input
    assert topk_idx.dim() == 3, "topk_idx must be 3D: (batch_size, seq_len, num_experts_per_tok)"
    # Flatten to 1D and make contiguous int32
    flat = topk_idx.reshape(-1).contiguous()
    flat32 = flat.to(torch.int32)
    N = flat32.numel()
    # Counts buffer: int32, length NUM_CLASSES = 256
    counts = torch.zeros(256, dtype=torch.int32, device=flat32.device)
    # Scan buffer: int32, length NUM_CLASSES + 1 = 257
    scan = torch.empty(257, dtype=torch.int32, device=flat32.device)

    # Output for sorted token indices: int64, length N
    out_idx = torch.empty(N, dtype=torch.int64, device=flat32.device)

    # Launch the single Triton kernel. Grid = (1,) since we do serial loops inside the kernel.
    _hist_and_offsets_kernel[(1,)](flat32, counts, scan, N, NUM_CLASSES=256)

    # sorted_token_indices is out_idx (already int64, shape (N,))
    sorted_token_indices = out_idx

    # expert_offsets length is NUM_CLASSES = 256 (return values without the leading zero)
    expert_offsets = scan[1:]  # int32, shape (256,)
    return sorted_token_indices, expert_offsets


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        # topk_idx: (batch_size, seq_len, num_experts_per_tok)
        sorted_token_indices, expert_offsets = _launch_hist_and_offsets(topk_idx)
        # sorted_token_indices: 1D int64, length N
        # expert_offsets: 1D int32, length 256
        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
