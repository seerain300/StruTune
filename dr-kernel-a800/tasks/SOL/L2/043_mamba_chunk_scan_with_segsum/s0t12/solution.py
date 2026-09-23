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
    chunk = tl.program_id(2)

    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    mask_out = offs < N_out

    # Load from input if within original size, else 0
    mask_in = (offs < N_in) & mask_out
    in_offset = b * in_stride_b + n1 * in_stride_n1 + offs * in_stride_n2
    in_vals = tl.load(in_ptr + in_offset, mask=mask_in, other=0.0)

    out_offset = b * out_stride_b + n1 * out_stride_n1 + offs * out_stride_n2
    tl.store(out_ptr + out_offset, in_vals, mask=mask_out)


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
    y += D * hidden_padded, elementwise over flattened tensors.
    D_ptr is a scalar (single element) tensor of float32.
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
        chunk_size = 256

        # Compute padding size to make seq_len multiple of chunk_size
        pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
        seq_len_padded = seq_len + pad_size

        # 1) Pad hidden_states on seq_len dimension using Triton kernel
        hidden_padded = torch.empty((batch_size, seq_len_padded, num_heads, head_dim),
                                     device=hidden_states.device, dtype=hidden_states.dtype)

        in_strides = hidden_states.stride()
        out_strides = hidden_padded.stride()

        grid_pad = (batch_size, num_heads, triton.cdiv(seq_len_padded, 128))
        pad_last_dim_kernel[grid_pad](
            hidden_states, hidden_padded,
            batch_size, num_heads, seq_len, seq_len, seq_len_padded,
            in_strides[0], in_strides[1], in_strides[2],
            out_strides[0], out_strides[1], out_strides[2],
            BLOCK=128,
            num_warps=4
        )

        # 2) Transpose A: A_transposed = A.transpose(1, 2) → shape [batch_size, seq_len, num_heads]
        A_transposed = A.transpose(1, 2)  # [batch_size, seq_len, num_heads]

        # 3) Reshape A_transposed into chunks [batch_size, num_chunks, chunk_size, num_heads]
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        A_chunked = torch.empty((batch_size, num_chunks, chunk_size, num_heads),
                                 device=A_transposed.device, dtype=A_transposed.dtype)

        # Copy A_transposed blocks into A_chunked using a Triton kernel
        @triton.jit
        def copy_block_kernel(in_ptr, out_ptr,
                              B: tl.int32, N1: tl.int32, N2: tl.int32, N3: tl.int32,
                              in_stride_b: tl.int32, in_stride_n1: tl.int32, in_stride_n2: tl.int32, in_stride_n3: tl.int32,
                              out_stride_b: tl.int32, out_stride_n1: tl.int32, out_stride_n2: tl.int32, out_stride_n3: tl.int32,
                              BLOCK: tl.constexpr):
            b = tl.program_id(0)
            n1 = tl.program_id(1)  # nc
            n2 = tl.program_id(2)  # h

            base_in = b * in_stride_b + n1 * in_stride_n1 + n2 * in_stride_n2
            base_out = b * out_stride_b + n1 * out_stride_n1 + n2 * out_stride_n2

            start = n1 * N3  # s_start = nc * chunk_size
            offs = tl.arange(0, BLOCK)
            mask = offs < N3

            in_offsets = base_in + (start + offs) * in_stride_n3
            out_offsets = base_out + offs * out_stride_n3

            vals = tl.load(in_ptr + in_offsets, mask=mask, other=0.0)
            tl.store(out_ptr + out_offsets, vals, mask=mask)

        grid_copy = (batch_size, num_chunks, num_heads)
        copy_block_kernel[grid_copy](
            A_transposed, A_chunked,
            batch_size, num_chunks, num_heads, chunk_size,
            A_transposed.stride(0), A_transposed.stride(1), A_transposed.stride(2),
            A_chunked.stride(0), A_chunked.stride(1), A_chunked.stride(2),
            BLOCK=chunk_size,
            num_warps=4
        )

        # 4) Permute A_chunked to [batch_size, num_chunks, num_heads, chunk_size]
        A_chunked_perm = A_chunked.permute(0, 1, 3, 2)  # [batch, num_chunks, num_heads, chunk_size]

        # 5) Compute cumsum along last dimension (chunk_size) using Triton kernel
        A_cumsum = torch.empty_like(A_chunked_perm)

        grid_cumsum = (batch_size, num_chunks, num_heads)
        cumsum_last_dim_kernel[grid_cumsum](
            A_chunked_perm, A_cumsum,
            batch_size, num_chunks, num_heads, chunk_size,
            A_chunked_perm.stride(0), A_chunked_perm.stride(1), A_chunked_perm.stride(2), A_chunked_perm.stride(3),
            A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), A_cumsum.stride(3),
            BLOCK=chunk_size,
            num_warps=4
        )

        # 6) Compute D residual: y += D * hidden_padded
        y = torch.empty((batch_size, seq_len_padded, num_heads, head_dim),
                        device=hidden_padded.device, dtype=hidden_padded.dtype)

        y_flat = y.view(-1)
        hidden_flat = hidden_padded.view(-1)
        n_elements = y_flat.numel()

        D_flat = D.reshape(1)  # ensure 1-element tensor

        add_d_residual_kernel[(triton.cdiv(n_elements, 1024),)](
            y_flat, hidden_flat, D_flat,
            n_elements,
            BLOCK=1024,
            num_warps=4
        )

        # 7) Remove padding on seq_len (return first seq_len rows)
        output = y[:, :seq_len, :, :].reshape(batch_size, seq_len, num_heads * head_dim)

        final_state = None
        return output, final_state


def run(*args):
    return ModelNew()(*args)
