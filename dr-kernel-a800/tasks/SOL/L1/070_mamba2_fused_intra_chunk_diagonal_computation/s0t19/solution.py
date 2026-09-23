import torch
import triton
import triton.language as tl


@triton.jit
def build_L_kernel(
    A_ptr,                  # *f32, shape [B, H, C, S]
    L_ptr,                  # *f32, shape [B, C, S, S, H]
    Bsz, Csz, Hsz, S,       # int32
    stride_ab, stride_ab_h, stride_ab_c, stride_ab_s,   # int64
    stride_al_b, stride_al_c, stride_al_i, stride_al_j, stride_al_h,  # int64
):
    # One program per (b, c, h) triple
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)

    # Bounds check (in case grid > dimensions)
    if (b >= Bsz) or (c >= Csz) or (h >= Hsz):
        return

    # Compute cumsum along S for A[b, h, c, :]
    cumsum = tl.zeros([1], dtype=tl.float32)
    for j in range(0, S):
        a = tl.load(A_ptr + b * stride_ab + h * stride_ab_h + c * stride_ab_c + j * stride_ab_s)
        cumsum += a
        # store L[i, j, h] for all i >= j
        for i in range(j, S):
            tl.store(L_ptr + b * stride_al_b + c * stride_al_c + i * stride_al_i + j * stride_al_j + h * stride_al_h, tl.exp(cumsum))


@triton.jit
def compute_Y_diag_kernel(
    L_ptr,                  # *f32, shape [B, C, S, S, H]
    hidden_ptr,             # *f32, shape [B, C, S, H, head_dim]
    Y_ptr,                  # *f32, shape [B, C, S, H, head_dim]
    Bsz, Csz, S, Hsz, Hdim, # int32
    stride_lb, stride_lc, stride_li, stride_lj, stride_lh,     # int64
    stride_hb, stride_hc, stride_hs, stride_hh, stride_hd,     # int64
    stride_yb, stride_yc, stride_yi, stride_yh, stride_yd,     # int64
):
    # One program per (b, c, i, h, d)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    d = tl.program_id(4)

    if (b >= Bsz) or (c >= Csz) or (i >= S) or (h >= Hsz) or (d >= Hdim):
        return

    acc = tl.zeros([1], dtype=tl.float32)
    for j in range(0, S):
        # Load L[b, c, i, j, h]
        l_val = tl.load(L_ptr + b * stride_lb + c * stride_lc + i * stride_li + j * stride_lj + h * stride_lh)
        # Load hidden[b, c, j, h, d]
        h_val = tl.load(hidden_ptr + b * stride_hb + c * stride_hc + j * stride_hs + h * stride_hh + d * stride_hd)
        acc += l_val * h_val

    # Store Y[b, c, i, h, d] (will cast to bfloat16 in host if needed)
    tl.store(Y_ptr + b * stride_yb + c * stride_yc + i * stride_yi + h * stride_yh + d * stride_yd, acc)


class ModelNew(torch.nn.Module):
    def __init__(self, chunk_size: int = 128, num_heads: int = 32, n_groups: int = 8):
        super().__init__()
        self.chunk_size = chunk_size
        self.num_heads = num_heads
        self.n_groups = n_groups  # not used, kept for signature compatibility

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Compute output similar to the original run function, using Triton kernels.
        - hidden_states: [B, C, S, H, head_dim]
        - A_cumsum: [B, H, C, S] (float32 or float16; we'll cast to float32 for compute)
        """
        # Dimensions
        Bsz, Csz, S, Hsz, Hdim = hidden_states.shape
        assert S == self.chunk_size, f"hidden_states last dim must be chunk_size={self.chunk_size}, got {S}"
        assert Hsz == self.num_heads, f"hidden_states dim H must be num_heads={self.num_heads}, got {Hsz}"
        assert A_cumsum.shape == (Bsz, Hsz, Csz, S), f"A_cumsum shape must be [B, H, C, S], got {A_cumsum.shape}"

        # Ensure contiguous tensors and dtype float32 for compute
        A = A_cumsum.to(torch.float32).contiguous()
        hidden = hidden_states.to(torch.float32).contiguous()

        # Allocate L and Y (float32 for computation)
        L = torch.empty((Bsz, Csz, S, S, Hsz), dtype=torch.float32, device=hidden.device)
        Y = torch.empty((Bsz, Csz, S, Hsz, Hdim), dtype=torch.float32, device=hidden.device)

        # Launch kernel to build L
        grid_L = (Bsz, Csz, Hsz)
        build_L_kernel[grid_L](
            A, L,
            Bsz, Csz, Hsz, S,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
        )

        # Launch kernel to compute Y_diag
        grid_Y = (Bsz, Csz, S, Hsz, Hdim)
        compute_Y_diag_kernel[grid_Y](
            L, hidden, Y,
            Bsz, Csz, S, Hsz, Hdim,
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            hidden.stride(0), hidden.stride(1), hidden.stride(2), hidden.stride(3), hidden.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
        )

        # Cast output to bfloat16 to match original run output dtype
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
