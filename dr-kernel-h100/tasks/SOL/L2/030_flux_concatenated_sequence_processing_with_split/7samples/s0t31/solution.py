import torch
import triton
import triton.language as tl


@triton.jit
def concat_seq_dim1_kernel(
    out_ptr,          # *fp32, output A: [B, M, K], M = T + I
    in1_ptr,          # *fp32, encoder_hidden_states: [B, T, K]
    in2_ptr,          # *fp32, hidden_states: [B, I, K]
    B, T, I, K,       # sizes (runtime ints)
    OUT_s0, OUT_s1, OUT_s2,   # strides for A
    IN1_s0, IN1_s1, IN1_s2,   # strides for encoder_hidden_states
    IN2_s0, IN2_s1, IN2_s2,   # strides for hidden_states
    BLOCK_M: tl.constexpr,    # tile along sequence dim (M)
    BLOCK_K: tl.constexpr,    # tile along hidden dim (K)
):
    # 3D launch grid: (B, tiles over M, tiles over K)
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    k_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)  # [BLOCK_K]

    mask_m = m_offsets < (T + I)
    mask_k = k_offsets < K

    # Determine which input to read from: rows < T come from in1, else from in2
    is_encoder = m_offsets < T  # boolean mask per m

    # Compute destination pointers for A[b, m, k]
    # A is [B, M, K] with strides OUT_s0, OUT_s1, OUT_s2
    out_base = b * OUT_s0
    out_ptrs = out_ptr + out_base + m_offsets[:, None] * OUT_s1 + k_offsets[None, :] * OUT_s2  # [BLOCK_M, BLOCK_K]

    # Compute source pointers
    # For encoder rows: in1[b, m, k]
    in1_base = b * IN1_s0
    in1_ptrs = in1_ptr + in1_base + m_offsets[:, None] * IN1_s1 + k_offsets[None, :] * IN1_s2  # [BLOCK_M, BLOCK_K]
    # For image rows: in2[b, (m - T), k]
    in2_base = b * IN2_s0
    in2_ptrs = in2_ptr + in2_base + (m_offsets[:, None] - T) * IN2_s1 + k_offsets[None, :] * IN2_s2  # [BLOCK_M, BLOCK_K]

    # Select source based on is_encoder
    # We need a 2D [BLOCK_M, BLOCK_K] pointer
    src_ptrs = tl.where(is_encoder[:, None], in1_ptrs, in2_ptrs)  # broadcast boolean along K

    # Combine masks for load/store
    load_mask = mask_m[:, None] & mask_k[None, :]
    store_mask = load_mask

    # Load and store
    vals = tl.load(src_ptrs, mask=load_mask, other=0.0)
    tl.store(out_ptrs, vals, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run:
        1) Concatenate encoder_hidden_states and hidden_states along sequence dimension (dim=1) via Triton.
        2) Compute processed = concatenated @ process_weight.T using torch.matmul (no bias).
        3) Split processed back into encoder and image streams and return.

        Returns:
            Tuple of (processed_encoder_hidden_states, processed_hidden_states)
        """
        # Sizes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        K = hidden_states.shape[2]
        M = T + I

        # Ensure inputs are contiguous for simpler stride handling
        encoder_hidden_states = encoder_hidden_states.contiguous()
        hidden_states = hidden_states.contiguous()
        process_weight = process_weight.contiguous()  # [K, K]

        # Allocate output A for concatenation: [B, M, K]
        A = torch.empty((B, M, K), device=hidden_states.device, dtype=torch.float32)

        # Launch Triton concat kernel
        BLOCK_M = 128
        BLOCK_K = 64
        grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
        concat_seq_dim1_kernel[grid](
            A, encoder_hidden_states, hidden_states,
            B, T, I, K,
            A.stride(0), A.stride(1), A.stride(2),
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Matmul: processed = A @ process_weight
        # A is [B, M, K], process_weight is [K, K] -> output [B, M, K]
        W = process_weight  # already [K, K]; use as is
        # torch.matmul handles broadcasting over batch
        C = torch.matmul(A, W)  # [B, M, K]

        # Split back into two streams
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        # Cast back to original dtypes to match inputs
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
