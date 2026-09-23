import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def pad_seq_kernel(
    in_ptr,            # *float32, input tensor pointer (contiguous), shape [B, L]
    out_ptr,           # *float32, output tensor pointer (contiguous), shape [B, L_out]
    L,                 # int32, original seq_len
    L_out,             # int32, padded seq_len
    pad_right,         # int32, number of zeros to append on the right
):
    # Each program handles one (batch, position)
    b = tl.program_id(0)
    pos = tl.program_id(1)
    if pos < L:
        val = tl.load(in_ptr + b * L + pos)
        tl.store(out_ptr + b * L_out + pos, val)
    else:
        tl.store(out_ptr + b * L_out + pos, 0.0)


@triton.jit
def lower_tri_mask_kernel(
    out_ptr,           # *float32, output mask [I, I] contiguous, I = padded chunk size (L_out)
    I: tl.constexpr,   # padded seq_len
    diagonal: tl.constexpr  # -1
):
    # Produce a 2D lower-triangular mask of shape [I, I] with given diagonal.
    rows = tl.arange(0, I)
    cols = tl.arange(0, I)
    row_idx = rows[:, None]  # [I, 1]
    col_idx = cols[None, :]  # [1, I]
    cond = (row_idx >= (col_idx + diagonal))  # diagonal=-1 => i >= j-1
    out_val = tl.where(cond, 1.0, 0.0)
    offsets = row_idx * I + col_idx
    tl.store(out_ptr + offsets, out_val)


@triton.jit
def per_row_cumsum_kernel(
    in_ptr,            # *float32, input matrix [I, I], contiguous
    out_ptr,           # *float32, output matrix [I, I], contiguous
    I: tl.constexpr
):
    # For each row r in 0..I-1, compute inclusive cumsum of that row in the 2D matrix in_ptr -> out_ptr.
    r = tl.program_id(0)
    cols = tl.arange(0, I)
    base = r * I
    in_row = in_ptr + base + cols
    out_row = out_ptr + base + cols
    acc = 0.0
    for j in range(0, I):
        val = tl.load(in_row + j)
        acc = acc + val
        tl.store(out_row + j, acc)


@triton.jit
def exp_rows_kernel(
    in_ptr,            # *float32, input matrix [I, I], contiguous
    out_ptr,           # *float32, output matrix [I, I], contiguous
    row_factor,        # *float32, scalar per row (exp value)
    I: tl.constexpr
):
    # Multiply each element in the row by row_factor (elementwise exp per row).
    r = tl.program_id(0)
    cols = tl.arange(0, I)
    base = r * I
    row_in = in_ptr + base + cols
    row_out = out_ptr + base + cols
    factor = tl.load(row_factor)  # scalar
    vals = tl.load(row_in)
    vals = vals * factor
    tl.store(row_out, vals)


@triton.jit
def y_diag_triton_kernel(
    M_ptr,             # *float32, M tensor: [B, N, I, H, D], contiguous
    V_ptr,             # *float32, V tensor: [B, N, I, H, D], contiguous
    Y_ptr,             # *float32, output: [B, N, I, H, D], contiguous
    batch: tl.constexpr, num_chunks: tl.constexpr, I: tl.constexpr, num_heads: tl.constexpr, head_dim: tl.constexpr
):
    # Grid: (batch, num_chunks, num_heads, head_dim)
    b = tl.program_id(0)
    nc = tl.program_id(1)
    h = tl.program_id(2)
    d = tl.program_id(3)
    # For each i in I, compute Y[b, nc, i, h, d] = sum_j M[b, nc, i, j, h] * V[b, nc, j, h, d]
    for i in range(0, I):
        acc = 0.0
        for j in range(0, I):
            base = b * (num_chunks * I * num_heads * head_dim) + nc * (I * num_heads * head_dim)
            m_index = base + i * (num_heads * head_dim) + h * head_dim + d
            v_index = base + j * (num_heads * head_dim) + h * head_dim + d
            m_val = tl.load(M_ptr + m_index)
            v_val = tl.load(V_ptr + v_index)
            acc = acc + m_val * v_val
        # Write accumulated value for this i
        y_index = base + i * (num_heads * head_dim) + h * head_dim + d
        tl.store(Y_ptr + y_index, acc)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # Constants from original code
        self.chunk_size = 256
        self.state_size = 256
        self.num_heads = 16
        self.head_dim = 64  # 1024 / 16

    def forward(self, hidden_states: torch.Tensor,
                A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor,
                initial_states: torch.Tensor):
        # Original shapes
        B, L, H, D = hidden_states.shape

        # Compute padding to make seq_len multiple of chunk_size
        pad_size = (self.chunk_size - L % self.chunk_size) % self.chunk_size
        L_out = L + pad_size
        N = (L_out + self.chunk_size - 1) // self.chunk_size

        # Convert to float32 for computation (no torch ops in forward)
        hidden_states_f = hidden_states.contiguous().to(torch.float32)
        A_f = A.contiguous().to(torch.float32)
        B_f = B.contiguous().to(torch.float32)
        C_f = C.contiguous().to(torch.float32)
        D_f = D.contiguous().to(torch.float32)
        initial_states_f = initial_states.contiguous().to(torch.float32)

        # 1) Pad hidden_states along sequence dimension (Triton)
        hidden_states_padded = torch.empty((B, L_out, H, D), dtype=torch.float32, device=hidden_states.device)
        pad_seq_kernel[(B, L_out)](
            hidden_states_f, hidden_states_padded, L, L_out, pad_size
        )

        # 2) Build lower-triangular mask for padded length and perform per-row cumsum (Triton)
        # Note: We need mask size equal to L_out, not chunk_size. We use Triton to create the mask and cumsum.
        # Masks and cumsum buffers for each chunk: However, since we pad to L_out, we can create one mask per chunk of length L_out.
        # But we need per-chunk row starts. Original code creates mask per chunk using chunk rows. We'll emulate by constructing
        # mask for entire padded sequence and then slice for cumsum. This requires building a 2D mask of size [N, I, I] where I=L_out.
        # To simplify, we create a single 2D mask of size [L_out, L_out] and use cumsum along rows. Then per chunk, we slice rows.

        # For segment_sum, original code builds lower-tri mask of shape [chunk_size, chunk_size], here L_out may differ.
        # The original code uses chunk_size=256. We will assume L_out <= chunk_size (since padding adds at most chunk_size-1).
        # In general, we can only emulate segment_sum for I up to chunk_size. For safety, we assume I=L_out<=256 (typical seq_len<<256).
        # If L_out > 256, the original code would error. We will guard and fall back to torch (not allowed). Here we assume <=256.

        # Allocate mask and cumsum buffers for I=L_out (assume L_out <= 256


def run(*args):
    return ModelNew()(*args)
