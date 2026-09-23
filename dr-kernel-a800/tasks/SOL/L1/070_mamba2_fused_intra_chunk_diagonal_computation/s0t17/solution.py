import torch
import triton
import triton.language as tl

# Kernel 1: build L[b, c, i, j, h] = exp(cumsum(A[b, h, c, :])[j]) for i >= j
@triton.jit
def build_L_from_Acumsum_kernel(
    A_ptr,          # *float32, shape [B, H, C, S]
    L_ptr,          # *float32, shape [B, C, S, S, H]
    Bsz, Csz, Hsz, Ssz,
    stride_Ab, stride_Ah, stride_Ac, stride_Aj,  # strides for A
    stride_Lb, stride_Lc, stride_Li, stride_Lj, stride_Lh,  # strides for L
):
    # Each program handles one (b, c, h) triple and writes L for all i, j
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)

    # Initialize cumsum vector of size Ssz
    # Triton supports scalar/vector ops; we use a vector of 128 elements
    # but since we loop j from 0..Ssz-1, we maintain a vector of current sum.
    cumsum = tl.zeros([Ssz], dtype=tl.float32)

    # We will write L[i, j, h] = exp(cumsum[j]) for i >= j
    # We use nested loops over i and j; since Triton programs are 1D, we rely on masks.
    # However, Triton doesn't support 4D nested loops cleanly in 1D kernels; we instead
    # write all i, j pairs by indexing outer i and j ranges. To keep it simple and robust,
    # we pre-fill L with zeros and then write only the lower-triangular part by launching
    # over i in [j..Ssz-1]. To achieve that, we do two passes: first fill upper with 0,
    # then write lower-triangular. Here we write lower-triangular only.
    # But Triton doesn't have such fill easily; so we write for all i, j with i >= j.

    # Pre-fill L with zeros (host does it before kernel launch), then overwrite lower-triangular
    # We implement overwrite by computing cumsum and writing L[i, j, h] for i >= j.
    # Note: Triton programs here are (b, c, h), and we loop j and i manually.
    # We'll iterate j first, then i >= j.

    # Loop j from 0 to Ssz-1
    for j in range(0, Ssz):
        # Load A[b, h, c, j]
        # Address: b*stride_Ab + h*stride_Ah + c*stride_Ac + j*stride_Aj
        a_val = tl.load(A_ptr + b * stride_Ab + h * stride_Ah + c * stride_Ac + j * stride_Aj)
        # Update cumsum
        cumsum = cumsum + a_val
        # Exponential value for this j
        exp_val = tl.exp(cumsum)
        # Now write L[i, j, h] for i >= j
        # We need i in [j, Ssz-1]; we can iterate i and write one by one.
        # Triton allows scalar loop via range, but writing 2D via scalar is fine for Ssz=128.
        for i in range(j, Ssz):
            # Store to L[b, c, i, j, h]
            tl.store(L_ptr + b * stride_Lb + c * stride_Lc + i * stride_Li + j * stride_Lj + h * stride_Lh, exp_val)

# Kernel 2: compute Y_diag[b, c, i, h, d] = sum_j L[b, c, i, j, h] * hidden_states[b, c, j, h, d]
@triton.jit
def compute_Y_diag_kernel(
    L_ptr,          # *float32, shape [B, C, S, S, H]
    hidden_ptr,     # *float32, shape [B, C, S, H, D]
    Y_ptr,          # *float32, shape [B, C, S, H, D]
    Bsz, Csz, Ssz, Hsz, Dsz,
    stride_Lb, stride_Lc, stride_Li, stride_Lj, stride_Lh,
    stride_hb, stride_hc, stride_hs, stride_hh, stride_hd,
    stride_Yb, stride_Yc, stride_Yi, stride_Yh, stride_Yd,
):
    # Each program handles one output element (b, c, i, h, d)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    d = tl.program_id(4)

    acc = tl.zeros((), dtype=tl.float32)
    # Loop over j and accumulate
    for j in range(0, Ssz):
        # Load L[b, c, i, j, h]
        L_val = tl.load(L_ptr + b * stride_Lb + c * stride_Lc + i * stride_Li + j * stride_Lj + h * stride_Lh)
        # Load hidden_states[b, c, j, h, d]
        hidden_val = tl.load(hidden_ptr + b * stride_hb + c * stride_hc + j * stride_hs + h * stride_hh + d * stride_hd)
        acc += L_val * hidden_val

    # Store result
    tl.store(Y_ptr + b * stride_Yb + c * stride_Yc + i * stride_Yi + h * stride_Yh + d * stride_Yd, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants implied by original code
        self.CHUNK_SIZE = 128
        self.NUM_HEADS = 32

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Compute Y_diag as in the original run function, but entirely with Triton kernels.
        Inputs:
          - hidden_states: [B, C, S, H, head_dim]
          - A_cumsum:      [B, H, C, S]
          - B, C: not used (original code doesn't use them)
        Output:
          - Y_diag: [B, C, S, H, head_dim] in bfloat16
        """
        # Ensure inputs are on CUDA and contiguous
        assert hidden_states.is_cuda and A_cumsum.is_cuda, "Inputs must be CUDA tensors."
        Bsz, Csz, Ssz, Hsz, Dsz = hidden_states.shape
        # A_cumsum shape: [B, H, C, S]
        assert A_cumsum.shape == (Bsz, Hsz, Csz, Ssz), "A_cumsum must have shape [B, H, C, S]."

        # Allocate L and Y (float32 for computation)
        L = torch.empty((Bsz, Csz, Ssz, Ssz, Hsz), device=hidden_states.device, dtype=torch.float32)
        Y = torch.empty((Bsz, Csz, Ssz, Hsz, Dsz), device=hidden_states.device, dtype=torch.float32)

        # Launch kernel to build L from A_cumsum
        # Grid: (B, C, H)
        grid_L = (Bsz, Csz, Hsz)
        build_L_from_Acumsum_kernel[grid_L](
            A_cumsum, L,
            Bsz, Csz, Hsz, Ssz,
            A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), A_cumsum.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            num_warps=1, num_stages=1
        )

        # Launch kernel to compute Y_diag
        grid_Y = (Bsz, Csz, Ssz, Hsz, Dsz)
        compute_Y_diag_kernel[grid_Y](
            L, hidden_states, Y,
            Bsz, Csz, Ssz, Hsz, Dsz,
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2), hidden_states.stride(3), hidden_states.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
            num_warps=1, num_stages=1
        )

        # Cast to bfloat16 to match original run output dtype
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
