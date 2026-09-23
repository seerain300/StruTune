import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def pad_seq_kernel(
    in_ptr,            # *float32, input tensor pointer (contiguous), shape [B, L]
    out_ptr,           # *float32, output tensor pointer (contiguous), shape [B, L_out]
    L: tl.constexpr,   # original seq_len
    L_out: tl.constexpr,  # padded seq_len
    pad_right: tl.constexpr  # number of zeros to append on the right
):
    # Each program handles one (batch, position)
    b = tl.program_id(0)
    pos = tl.program_id(1)
    if pos < L:
        val = tl.load(in_ptr + b * L + pos)
        tl.store(out_ptr + b * L_out + pos, val)
    else:
        # Write zeros for padded positions
        tl.store(out_ptr + b * L_out + pos, 0.0)


@triton.jit
def lower_tri_mask_kernel(
    out_ptr,           # *float32, output mask [I, I] contiguous, I = chunk_size
    I: tl.constexpr,   # chunk_size
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
def per_row_cumsum_kernel(in_ptr, out_ptr, chunk_size: tl.constexpr):
    # For each row r in 0..chunk_size-1, compute inclusive cumsum of that row in the 2D matrix in_ptr -> out_ptr.
    # in_ptr and out_ptr are contiguous [chunk_size, chunk_size].
    r = tl.program_id(0)
    cols = tl.arange(0, chunk_size)
    base = r * chunk_size
    in_row = in_ptr + base + cols
    out_row = out_ptr + base + cols
    acc = 0.0
    for j in range(0, chunk_size):
        val = tl.load(in_row + j)
        acc = acc + val
        tl.store(out_row + j, acc)


@triton.jit
def y_diag_triton_kernel(
    M_ptr,  # M tensor: [B, N, I, H, D], contiguous
    V_ptr,  # V tensor: [B, N, I, H, D], contiguous
    Y_ptr,  # output: [B, N, I, H, D], contiguous
    batch: tl.constexpr, num_chunks: tl.constexpr, chunk_size: tl.constexpr, num_heads: tl.constexpr, head_dim: tl.constexpr
):
    # Grid: (batch, num_chunks, num_heads, head_dim)
    b = tl.program_id(0)
    nc = tl.program_id(1)
    h = tl.program_id(2)
    d = tl.program_id(3)
    # For each i in chunk_size, compute Y[b, nc, i, h, d] = sum_j M[b, nc, i, j, h] * V[b, nc, j, h, d]
    for i in range(0, chunk_size):
        acc = 0.0
        for j in range(0, chunk_size):
            # Compute flat indices assuming layout [B, N, I, H, D] contiguous.
            # M is laid out as [B, N, I, H, D] contiguous:
            # idx = b * (N*I*H*D) + nc * (I*H*D) + i * (H*D) + h * D + d
            M_idx = (b * (num_chunks * chunk_size * num_heads * head_dim)) + \
                    (nc * (chunk_size * num_heads * head_dim)) + \
                    (i * (num_heads * head_dim)) + \
                    (h * head_dim) + d
            V_idx = (b * (num_chunks * chunk_size * num_heads * head_dim)) + \
                    (nc * (chunk_size * num_heads * head_dim)) + \
                    (j * (num_heads * head_dim)) + \
                    (h * head_dim) + d
            m_val = tl.load(M_ptr + M_idx)
            v_val = tl.load(V_ptr + V_idx)
            acc = acc + m_val * v_val
        # Store result for Y[b, nc, i, h, d]
        Y_idx = (b * (num_chunks * chunk_size * num_heads * head_dim)) + \
                (nc * (chunk_size * num_heads * head_dim)) + \
                (i * (num_heads * head_dim)) + \
                (h * head_dim) + d
        tl.store(Y_ptr + Y_idx, acc)


@triton.jit
def elementwise_exp_rows_kernel(in_ptr, out_ptr, rows_exp, I: tl.constexpr):
    # Multiply each row r of in_ptr by exp(rows_exp[r]). Both in_ptr and out_ptr are [I, I].
    r = tl.program_id(0)
    cols = tl.arange(0, I)
    base = r * I
    in_row = in_ptr + base + cols
    out_row = out_ptr + base + cols
    scale = tl.exp(rows_exp + r)
    for j in range(0, I):
        val = tl.load(in_row + j)
        val = val * scale
        tl.store(out_row + j, val)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor, initial_states: torch.Tensor):
        # Shapes
        batch_size, seq_len, num_heads, head_dim = hidden_states.shape
        state_size = 256
        chunk_size = 256

        # Compute padding size to make seq_len multiple of chunk_size
        pad_right = (chunk_size - seq_len % chunk_size) % chunk_size
        seq_len_padded = seq_len + pad_right

        # Allocate padded hidden states
        hidden_padded = torch.empty((batch_size, seq_len_padded), dtype=torch.float32, device=hidden_states.device)

        # Launch pad kernel for hidden states
        grid_pad_hs = (batch_size, seq_len_padded)
        pad_seq_kernel[grid_pad_hs](
            hidden_states.to(torch.float32).contiguous().view(-1), hidden_padded.view(-1),
            seq_len, seq_len_padded, pad_right
        )

        # Compute num_chunks
        num_chunks = (seq_len_padded + chunk_size - 1) // chunk_size

        # Prepare A_padded along sequence (same as hidden pad)
        # A is [B, S] in the original; we pad to seq_len_padded with zeros
        A_padded = torch.empty((batch_size, seq_len_padded), dtype=torch.float32, device=hidden_states.device)
        grid_pad_A = (batch_size, seq_len_padded)
        pad_seq_kernel[grid_pad_A](
            A.to(torch.float32).contiguous().view(-1), A_padded.view(-1),
            A.shape[1], seq_len_padded, (seq_len_padded - A.shape[1])
        )

        # Build A_perm: [B, N, I]
        A_perm = torch.empty((batch_size, num_chunks, chunk_size), dtype=torch.float32, device=hidden_states.device)
        # Fill per chunk: take A_padded[:, nc*chunk_size:(nc+1)*chunk_size], zero pad to chunk_size
        for nc in range(num_chunks):
            start = nc * chunk_size
            end = min(start + chunk_size, seq_len_padded)
            length = end - start
            chunk = A_padded[:, start:end].to(torch.float32).contiguous().view(batch_size, -1)
            if length < chunk_size:
                pad = chunk_size - length
                chunk = torch.cat([chunk, torch.zeros((batch_size, pad), dtype=torch.float32, device=hidden_states.device)])
            A_perm[:, nc, :] = chunk

        # Compute per-chunk cumsum along columns for A_perm: A_cumsum[:, :, :]
        A_cumsum = torch.empty((batch_size, num_chunks, chunk_size), dtype=torch.float32, device=hidden_states.device)
        grid_cumsum = (batch_size * num_chunks,)
        for b in range(batch_size):
            for nc in range(num_chunks):
                in_buf = A_perm[b, nc, :].contiguous()
                out_buf = A_cumsum[b, nc, :].contiguous()
                per_row_cumsum_kernel[(chunk_size,)](in_buf, out_buf, chunk_size)

        # Form L = exp(cumsum) per row. Triton elementwise_exp_rows_kernel: multiply each row by exp(row_sum).
        # Compute row sums of A_cumsum: row_sum[B, N] = sum over I
        row_sum = torch.empty((batch_size, num_chunks), dtype=torch.float32, device=hidden_states.device)
        for b in range(batch_size):
            for nc in range(num_chunks):
                acc = 0.0
                for j in range(chunk_size):
                    acc += A_cumsum[b, nc, j]
                row_sum[b, nc] = acc
        # Now L = exp(A_cumsum) * exp(row_sum)
        L = torch.empty((batch_size, num_chunks, chunk_size), dtype=torch.float32, device=hidden_states.device)
        for b in range(batch_size):
            for nc in range(num_chunks):
                scale = tl.exp(row_sum[b, nc])  # Triton not used here in forward; keep torch for L formation
                L[b, nc, :] = torch.exp(A_cumsum[b, nc, :]) * scale

        # Compute segment masks for each chunk: lower-triangular with diagonal=-1
        mask_buf = torch.empty((num_chunks, chunk_size, chunk_size), dtype=torch.float32, device=hidden_states.device)
        cumsum_buf = torch.empty((num_chunks, chunk_size, chunk_size), dtype=torch.float32, device=hidden_states.device)
        for nc in range(num_chunks):
            lower_tri_mask_kernel[(1,)](mask_buf[nc], chunk_size, -1)
            per_row_cumsum_kernel


def run(*args):
    return ModelNew()(*args)
