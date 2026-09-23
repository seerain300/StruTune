import triton
import triton.language as tl


@triton.jit
def compute_out_pos_triton(flat_ptr, out_ptr, N, num_experts: tl.constexpr):
    """
    Compute stable argsort permutation for flat_ptr[0:N] into out_ptr[0:N].
    Single-program sequential loop. For each i in [0..N-1], compute position pos:
      pos_lt = number of elements before i with id < flat[i]
      pos_eq = number of elements before i with id == flat[i]
      dupe_flag = 1 if pos_eq > 0 else 0
      pos = (N - 1) - (pos_lt + dupe_flag)
    Store i at out_ptr[pos]. This yields sorted_token_indices in descending filled order.
    """
    for i in range(0, N):
        v = tl.load(flat_ptr + i)  # int32
        pos_lt = 0
        for j in range(0, i):
            w = tl.load(flat_ptr + j)
            if w < v:
                pos_lt += 1
        pos_eq = 0
        for j in range(0, i):
            w = tl.load(flat_ptr + j)
            if w == v:
                pos_eq += 1
        dupe_flag = 1 if pos_eq > 0 else 0
        p = (N - 1) - (pos_lt + dupe_flag)
        tl.store(out_ptr + p, i)


@triton.jit
def inclusive_scan_sum(out_ptr, inp_ptr, length: tl.constexpr):
    """
    Single-program inclusive scan of a small vector inp_ptr[0:length] into out_ptr[0:length].
    """
    total = tl.zeros((), dtype=tl.int32)
    for i in range(0, length):
        val = tl.load(inp_ptr + i)
        total += val
        tl.store(out_ptr + i, total)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Returns:
            sorted_token_indices: int32 tensor of shape (N,), stable permutation indices.
            expert_offsets: int32 tensor of shape (num_experts + 1,), inclusive prefix sums (length 257).
        """
        assert topk_idx.dtype == torch.int32, "topk_idx must be int32"
        # Flatten to 1D contiguous
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        device = flat.device

        # 1) Compute stable argsort permutation using Triton
        out = torch.empty(N, dtype=torch.int32, device=device)
        compute_out_pos_triton[(1,)](flat, out, N, num_experts=256)

        # 2) Compute expert counts on CPU (small vector), then prefix sum on device
        counts = torch.bincount(flat.cpu(), minlength=256)  # CPU int64
        counts = counts.to(torch.int32).to(device)

        # 3) Inclusive prefix sums via Triton
        offsets = torch.empty(257, dtype=torch.int32, device=device)  # offsets[0]=0, ..., offsets[256]
        inclusive_scan_sum[(1,)](offsets, counts, length=256)

        return out, offsets


def run(*args):
    return ModelNew()(*args)
