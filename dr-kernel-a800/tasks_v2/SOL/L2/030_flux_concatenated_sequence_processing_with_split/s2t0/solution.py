import torch
import triton
import triton.language as tl

# Triton kernel: for each (batch, sequence position), compute C[b, m, :] = A_row @ B
# A_row is selected from either encoder_hidden_states or hidden_states depending on m < T.
@triton.jit
def _matmul_seq_kernel(
    A_ptr, B_ptr, C_ptr,
    B, L, T, N,
    # Strides for A: A can be either encoder or image depending on m < T; we pass strides for each.
    A_src_stride_b, A_src_stride_m, A_src_stride_k,  # for encoder
    A_img_stride_b, A_img_stride_m, A_img_stride_k,  # for image
    # Strides for B: [K, N] (process_weight.T)
    B_stride_k, B_stride_n,
    # Strides for C: [B, L, N]
    C_stride_b, C_stride_m, C_stride_n,
    BLOCK_N: tl.constexpr,  # tile over output hidden
    BLOCK_K: tl.constexpr,  # tile over reduction dim
):
    # program ids
    b = tl.program_id(0)  # batch
    m = tl.program_id(1)  # sequence position in [0, L)
    n_block = tl.program_id(2)  # output column block
    n_start = n_block * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    # Select source tensor for A_row based on whether m < T
    use_encoder = m < T

    # Accumulator [1, BLOCK_N] in fp32
    acc = tl.zeros((1, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    k0 = 0
    while k0 < N:  # N is the hidden_dim (K == N in this context)
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        if use_encoder:
            # A_row = A_ptr[b, m, k] from encoder
            A_row_ptrs = A_ptr + b * A_src_stride_b + m * A_src_stride_m + k_offsets * A_src_stride_k
        else:
            # A_row = A_ptr[b, m - T, k] from image
            A_row_ptrs = A_ptr + b * A_img_stride_b + (m - T) * A_img_stride_m + k_offsets * A_img_stride_k

        # B_block = B[k, n_offsets] -> [BLOCK_K, BLOCK_N]
        B_block_ptrs = B_ptr + k_offsets[:, None] * B_stride_k + n_offsets[None, :] * B_stride_n

        # Masks: ensure in-bounds
        a_mask = k_offsets < N
        b_mask = (k_offsets[:, None] < N) & (n_offsets[None, :] < N)

        # Load tiles; out-of-bounds get zeros
        A_row = tl.load(A_row_ptrs, mask=a_mask, other=0.0)  # shape [BLOCK_K]
        B_block = tl.load(B_block_ptrs, mask=b_mask, other=0.0)  # shape [BLOCK_K, BLOCK_N]

        # Accumulate: acc += A_row[:, None] * B_block
        acc += A_row[:, None] * B_block  # elementwise multiply, then sum over K

        k0 += BLOCK_K

    # Store result to C at (b, m, n_offsets)
    C_ptrs = C_ptr + b * C_stride_b + m * C_stride_m + n_offsets * C_stride_n
    c_mask = n_offsets < N
    tl.store(C_ptrs, acc, mask=c_mask)


def _launch_triton_seq_matmul(
    encoder: torch.Tensor,  # [B, T, H]
    hidden: torch.Tensor,   # [B, I, H]
    process_weight: torch.Tensor,  # [H, H]
    out: torch.Tensor,      # [B, L, H], allocated and zeroed
):
    """
    Launch Triton kernel to compute out = concat([encoder, hidden], dim=1) @ process_weight.T
    without explicitly concatenating the inputs. out must be allocated as float32.
    """
    assert encoder.is_cuda and hidden.is_cuda and process_weight.is_cuda and out.is_cuda
    B = encoder.shape[0]
    T = encoder.shape[1]
    I = hidden.shape[1]
    H = encoder.shape[2]
    assert hidden.shape[2] == H, "hidden and encoder must have same hidden_dim"
    assert process_weight.shape == (H, H), "process_weight must be [H, H]"

    # Prepare B = process_weight.T contiguous [H, H]
    B_mat = process_weight.t().contiguous()

    # Strides for encoder and hidden
    A_src_stride_b, A_src_stride_m, A_src_stride_k = encoder.stride(0), encoder.stride(1), encoder.stride(2)
    A_img_stride_b, A_img_stride_m, A_img_stride_k = hidden.stride(0), hidden.stride(1), hidden.stride(2)
    B_stride_k, B_stride_n = B_mat.stride(0), B_mat.stride(1)
    C_stride_b, C_stride_m, C_stride_n = out.stride(0), out.stride(1), out.stride(2)

    # Tile sizes
    BLOCK_N = 128 if H >= 128 else 64
    BLOCK_K = 64 if H >= 64 else 32

    # Grid: (batch, sequence positions, output blocks)
    L = T + I
    grid = (B, L, triton.cdiv(H, BLOCK_N))

    _matmul_seq_kernel[grid](
        encoder, hidden, B_mat, out,
        B, L, T, H,
        A_src_stride_b, A_src_stride_m, A_src_stride_k,
        A_img_stride_b, A_img_stride_m, A_img_stride_k,
        B_stride_k, B_stride_n,
        C_stride_b, C_stride_m, C_stride_n,
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version that:
          - Avoids explicit concatenation by computing per-sequence position
          - Applies linear projection via Triton batched matmul
          - Splits results back into encoder and image streams
        """
        # Ensure CUDA tensors
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors"
        # For numeric stability and Triton expectations, compute in float32
        H = encoder_hidden_states.shape[2]
        B = encoder_hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]

        # Allocate output [B, L, H] where L = T + I
        L = T + I
        out = torch.empty((B, L, H), device=hidden_states.device, dtype=torch.float32)

        # Launch Triton kernel
        _launch_triton_seq_matmul(encoder_hidden_states, hidden_states, process_weight, out)

        # Split outputs
        processed_encoder = out[:, :T, :]
        processed_hidden = out[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
