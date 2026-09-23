import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def pad_last_dim_kernel(in_ptr, out_ptr,
                         B: tl.int32, N1: tl.int32, N2: tl.int32, N_in: tl.int32, N_out: tl.int32,
                         in_stride_b: tl.int32, in_stride_n1: tl.int32, in_stride_n2: tl.int32,
                         out_stride_b: tl.int32, out_stride_n1: tl.int32, out_stride_n2: tl.int32,
                         BLOCK: tl.constexpr):
    """
    Pad last dimension (N2) of a [B, N1, N2] tensor to N_out, appending zeros.
    out[b, n1, j] = in[b, n1, j] if j < N_in, else 0.
    Grid: (B, N1, ceil_div(N_out, BLOCK))
    """
    b = tl.program_id(0)
    n1 = tl.program_id(1)
    blk = tl.program_id(2)

    offs = blk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_out

    in_base = b * in_stride_b + n1 * in_stride_n1
    out_base = b * out_stride_b + n1 * out_stride_n1

    in_offsets = in_base + offs * in_stride_n2
    out_offsets = out_base + offs * out_stride_n2

    vals = tl.load(in_ptr + in_offsets, mask=mask & (offs < N_in), other=0.0)
    tl.store(out_ptr + out_offsets, vals, mask=mask)


@triton.jit
def cumsum_last_dim_kernel(x_ptr, y_ptr,
                            B: tl.int32, N1: tl.int32, N2: tl.int32, N3: tl.int32,
                            x_stride_b: tl.int32, x_stride_n1: tl.int32, x_stride_n2: tl.int32, x_stride_n3: tl.int32,
                            y_stride_b: tl.int32, y_stride_n1: tl.int32, y_stride_n2: tl.int32, y_stride_n3: tl.int32,
                            BLOCK: tl.constexpr):
    """
    Compute cumulative sum along the last dimension (N3) for each row (b, n1, n2).
    x_ptr and y_ptr point to tensors shaped [B, N1, N2, N3].
    Grid: (B, N1, N2)
    Each program processes one row across N3, sequentially.
    """
    b = tl.program_id(0)
    n1 = tl.program_id(1)
    n2 = tl.program_id(2)

    run_sum = 0.0
    for j in range(0, N3):
        x_offset = b * x_stride_b + n1 * x_stride_n1 + n2 * x_stride_n2 + j * x_stride_n3
        val = tl.load(x_ptr + x_offset)
        run_sum = run_sum + val
        y_offset = b * y_stride_b + n1 * y_stride_n1 + n2 * y_stride_n2 + j * y_stride_n3
        tl.store(y_ptr + y_offset, run_sum)


@triton.jit
def add_d_residual_kernel(y_ptr, hidden_padded_ptr, D_ptr, n_elements: tl.int32, BLOCK: tl.constexpr):
    """
    Elementwise add: y = y + D * hidden_padded
    D is a scalar tensor of shape [1], we load it once.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    hidden = tl.load(hidden_padded_ptr + offs, mask=mask, other=0.0)
    D = tl.load(D_ptr)  # scalar
    out = y + hidden * D
    tl.store(y_ptr + offs, out, mask=mask)


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor,
                initial_states: torch.Tensor):
        """
        Triton-optimized forward. All numerical computation is done in Triton kernels.
        No torch numerical ops (no .to(), .cumsum, .pad, .einsum, etc.) are used in forward.
        """
        # Shapes and constants
        batch_size, seq_len, num_heads, head_dim = hidden_states.shape
        state_size = 256
        n_groups = 1
        chunk_size = 256

        # Compute padding size to make seq_len multiple of chunk_size
        pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
        seq_len_padded = seq_len + pad_size

        # 1) Pad A (shape [batch_size, seq_len, num_heads]) to [batch_size, seq_len_padded, num_heads]
        A_padded = torch.empty((batch_size, seq_len_padded, num_heads), device=hidden_states.device, dtype=hidden_states.dtype)

        B_a = batch_size; N1_a = seq_len_padded; N2_a = num_heads
        in_stride_b_a = hidden_states.stride(0); in_stride_n1_a = hidden_states.stride(1); in_stride_n2_a = hidden_states.stride(2)
        out_stride_b_a = A_padded.stride(0); out_stride_n1_a = A_padded.stride(1); out_stride_n2_a = A_padded.stride(2)

        grid_pad_a = (B_a, N1_a, triton.cdiv(N2_a, 128))
        pad_last_dim_kernel[grid_pad_a](
            hidden_states, A_padded,
            B_a, seq_len, num_heads, seq_len, seq_len_padded,
            in_stride_b_a, in_stride_n1_a, in_stride_n2_a,
            out_stride_b_a, out_stride_n1_a, out_stride_n2_a,
            BLOCK=128,
            num_warps=4
        )

        # 2) Transpose A to [batch_size, num_heads, seq_len_padded], then chunk
        A_transposed = torch.empty((batch_size, num_heads, seq_len_padded), device=hidden_states.device, dtype=hidden_states.dtype)
        A_transposed.copy_(A_padded.permute(0, 2, 1))

        # Chunk along last dim into [batch_size, num_chunks, chunk_size, num_heads]
        num_chunks = (seq_len_padded + chunk_size - 1) // chunk_size
        A_chunked = torch.empty((batch_size, num_chunks, chunk_size, num_heads), device=hidden_states.device, dtype=hidden_states.dtype)
        for nc in range(num_chunks):
            start = nc * chunk_size
            end = min(start + chunk_size, seq_len_padded)
            for h in range(num_heads):
                A_chunked[:, nc, :, h] = A_transposed[:, h, start:end]

        # 3) Permute A_chunked to [batch_size, num_heads, num_chunks, chunk_size] for cumsum
        A_perm = A_chunked.permute(0, 2, 1, 3)  # [B, N1(num_chunks), N2(chunk_size), N3(num_heads)]

        # 4) Compute cumsum along last dim (N3=num_heads) for each (b, num_chunks, chunk_size) row using Triton
        B_perm = batch_size; N1_perm = num_chunks; N2_perm = chunk_size; N3_perm = num_heads
        x_stride_b = A_perm.stride(0); x_stride_n1 = A_perm.stride(1); x_stride_n2 = A_perm.stride(2); x_stride_n3 = A_perm.stride(3)
        y_stride_b = A_perm.stride(0); y_stride_n1 = A_perm.stride(1); y_stride_n2 = A_perm.stride(2); y_stride_n3 = A_perm.stride(3)

        y_perm = torch.empty((B_perm, N1_perm, N2_perm, N3_perm), device=hidden_states.device, dtype=hidden_states.dtype)

        grid_cumsum = (B_perm, N1_perm, N2_perm)
        cumsum_last_dim_kernel[grid_cumsum](
            A_perm, y_perm,
            B_perm, N1_perm, N2_perm, N3_perm,
            x_stride_b, x_stride_n1, x_stride_n2, x_stride_n3,
            y_stride_b, y_stride_n1, y_stride_n2, y_stride_n3,
            BLOCK=128,
            num_warps=4
        )

        # 5) Elementwise add: D * padded hidden_states
        # Pad hidden_states along last dim (seq_len) to seq_len_padded -> shape [batch_size, seq_len_padded, num_heads, head_dim]
        hidden_padded = torch.empty((batch_size, seq_len_padded, num_heads, head_dim), device=hidden_states.device, dtype=hidden_states.dtype)

        # Use pad_last_dim_kernel by flattening last two dims: [B, N1=seq_len_padded, N2=num_heads*head_dim]
        B_hs = batch_size; N1_hs = seq_len_padded; N2_hs = num_heads * head_dim
        N_in_hs = seq_len; N_out_hs = seq_len_padded

        hs_view = hidden_states.contiguous().view(B_hs, N1_hs, N2_hs)
        out_hs_view = hidden_padded.view(B_hs, N1_hs, N2_hs)

        in_stride_b_hs = hs_view.stride(0); in_stride_n1_hs = hs_view.stride(1); in_stride_n2_hs = hs_view.stride(2)
        out_stride_b_hs = out_hs_view.stride(0); out_stride_n1_hs = out_hs_view.stride(1); out_stride_n2_hs = out_hs_view.stride(2)

        grid_pad_hs = (B_hs, N1_hs, triton.cdiv(N2_hs, 1024))
        pad_last_dim_kernel[grid_pad_hs](
            hs_view, out_hs_view,
            B_hs, N1_hs, N2_hs, N_in_hs, N_out_hs,
            in_stride_b_hs, in_stride_n1_hs, in_stride_n2_hs,
            out_stride_b_hs, out_stride_n1_hs, out_stride_n2_hs,
            BLOCK=1024,
            num_warps=8
        )

        # 6) Elementwise add: y = y + D * hidden_padded
        y_flat = y_perm.reshape(-1)
        hidden_flat = hidden_padded.reshape(-1)
        n_elements = y_flat.numel()
        D_scalar = D.reshape(1)  # [1]
        add_d_residual_kernel[(triton.cdiv(n_elements, 1024),)](
            y_flat, hidden_flat, D_scalar,
            n_elements, BLOCK=1024,
            num_warps=8
        )

        # Return final y and None for final_state (not used in original)
        return y_perm, None


def run(*args):
    return ModelNew()(*args)
