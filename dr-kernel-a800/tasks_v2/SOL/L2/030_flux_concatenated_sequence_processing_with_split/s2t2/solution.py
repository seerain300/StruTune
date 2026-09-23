import torch
import triton
import triton.language as tl


@triton.jit
def _seq_linear_kernel(
    encoder_ptr,   # [B, T, H]
    hidden_ptr,    # [B, I, H]
    B_ptr,         # [H, H] = process_weight.T
    C_ptr,         # [B, L, H] output
    B, L, T, H,
    encoder_stride_b, encoder_stride_m, encoder_stride_k,
    hidden_stride_b, hidden_stride_m, hidden_stride_k,
    B_stride_k, B_stride_n,
    C_stride_b, C_stride_m, C_stride_n,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (B, L, ceil_div(H, BLOCK_N))
    b = tl.program_id(0)
    m_total = tl.program_id(1)  # sequence position in [0, L)
    n_block = tl.program_id(2)

    # Compute output column offsets for this block
    n_start = n_block * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    # Determine source tensor for this sequence position
    use_encoder = m_total < T

    # Accumulator for this (b, m_total) row across BLOCK_N output columns
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Reduction over hidden_dim (K=H)
    k0 = 0
    while k0 < H:
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        if use_encoder:
            # Load A_row = encoder[b, m_total, k_offsets]
            A_row_ptrs = encoder_ptr + b * encoder_stride_b + m_total * encoder_stride_m + k_offsets * encoder_stride_k
        else:
            # Load A_row = hidden[b, m_total - T, k_offsets]
            m_img = m_total - T
            A_row_ptrs = hidden_ptr + b * hidden_stride_b + m_img * hidden_stride_m + k_offsets * hidden_stride_k

        # Load B_block = B[k_offsets, n_offsets] -> shape [BLOCK_K, BLOCK_N]
        B_block_ptrs = B_ptr + k_offsets[:, None] * B_stride_k + n_offsets[None, :] * B_stride_n

        # Masks for bounds
        a_mask = k_offsets < H
        b_mask = (k_offsets[:, None] < H) & (n_offsets[None, :] < H)

        # Loads
        A_row = tl.load(A_row_ptrs, mask=a_mask, other=0.0)                    # [BLOCK_K]
        B_block = tl.load(B_block_ptrs, mask=b_mask, other=0.0)               # [BLOCK_K, BLOCK_N]

        # Compute partial dot products and accumulate into acc
        # For each n in BLOCK_N, acc[n] += sum_k A_row[k] * B_block[k, n]
        # We can do this elementwise:
        for n_idx in range(BLOCK_N):
            # Note: B_block[:, n_idx] extracts column n_idx from the [BLOCK_K, BLOCK_N] tensor
            acc[n_idx] += tl.sum(A_row * B_block[:, n_idx], axis=0)

        k0 += BLOCK_K

    # Store the accumulated results to C[b, m_total, n_offsets]
    C_row_ptrs = C_ptr + b * C_stride_b + m_total * C_stride_m + n_offsets * C_stride_n
    c_mask = n_offsets < H
    tl.store(C_row_ptrs, acc, mask=c_mask)


def _launch_seq_linear(
    encoder_hidden_states: torch.Tensor,  # [B, T, H]
    hidden_states: torch.Tensor,         # [B, I, H]
    process_weight: torch.Tensor,        # [H, H]
    out: torch.Tensor,                   # [B, (T+I), H], float32, allocated by host
):
    """
    Triton kernel launch to compute:
      out[b, m, :] = (m < T ? encoder_hidden_states[b, m, :] : hidden_states[b, m - T, :]) @ process_weight.T
    without torch.cat or torch.matmul on host.
    """
    assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda and out.is_cuda
    B, T, H = encoder_hidden_states.shape
    B2, I, H2 = hidden_states.shape
    assert B == B2 and H == H2, "Encoder and hidden inputs must have same batch and hidden_dim"
    L = T + I

    # Ensure contiguous inputs
    encoder = encoder_hidden_states.contiguous()
    hidden = hidden_states.contiguous()
    B_mat = process_weight.t().contiguous()  # [H, H]

    # Strides
    encoder_stride_b, encoder_stride_m, encoder_stride_k = encoder.stride(0), encoder.stride(1), encoder.stride(2)
    hidden_stride_b, hidden_stride_m, hidden_stride_k = hidden.stride(0), hidden.stride(1), hidden.stride(2)
    B_stride_k, B_stride_n = B_mat.stride(0), B_mat.stride(1)
    C_stride_b, C_stride_m, C_stride_n = out.stride(0), out.stride(1), out.stride(2)

    # Tile sizes
    BLOCK_N = 128 if H >= 128 else 64
    BLOCK_K = 64 if H >= 64 else 32

    # Grid: cover all batch, all sequence positions, and output column blocks
    grid = (B, L, triton.cdiv(H, BLOCK_N))

    _seq_linear_kernel[grid](
        encoder, hidden, B_mat, out,
        B, L, T, H,
        encoder_stride_b, encoder_stride_m, encoder_stride_k,
        hidden_stride_b, hidden_stride_m, hidden_stride_k,
        B_stride_k, B_stride_n,
        C_stride_b, C_stride_m, C_stride_n,
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version that:
          - Avoids torch.cat and torch.matmul on host
          - Computes the linear projection using a Triton kernel that selects between encoder and hidden per sequence position
          - Splits the result back into encoder and hidden streams
        """
        # Ensure CUDA tensors; compute in float32 for stability
        device = hidden_states.device
        if hidden_states.dtype != torch.float32:
            hidden_states = hidden_states.float()
        if encoder_hidden_states.dtype != torch.float32:
            encoder_hidden_states = encoder_hidden_states.float()
        if process_weight.dtype != torch.float32:
            process_weight = process_weight.float()

        B = encoder_hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = encoder_hidden_states.shape[2]
        assert hidden_states.shape[2] == H, "hidden and encoder must have same hidden_dim"
        assert process_weight.shape == (H, H), "process_weight must be [H, H]"

        L = T + I

        # Allocate output [B, L, H], float32
        out = torch.empty((B, L, H), device=device, dtype=torch.float32)

        # Launch Triton kernel
        _launch_seq_linear(encoder_hidden_states, hidden_states, process_weight, out)

        # Split outputs
        processed_encoder = out[:, :T, :]
        processed_hidden = out[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
