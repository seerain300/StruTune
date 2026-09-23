import torch
import triton
import triton.language as tl


@triton.jit
def _batched_gemm_block_kernel(
    A_ptr,         # *fp32, input A: [B_total*M, D] (we pass concatenation or per-stream tensors)
    WT_ptr,        # *fp32, process_weight.T: [D, D]
    Out_ptr,       # *fp32, output: [B_total, M, D]
    B_total: tl.constexpr,  # number of batches (we pass B for both streams; grid handles modulo)
    M,              # sequence length for this stream
    D,              # hidden_dim
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 3D launch grid: (B, tiles over M, tiles over N)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    b = pid_b  # pid_b in [0, B_total)
    # Compute tile indices
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Masks for valid m,n
    mask_m = m_offsets < M
    mask_n = n_offsets < D

    # Create accumulator [BLOCK_M, BLOCK_N] in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K (hidden_dim) in chunks
    # We iterate kk from 0 to D in steps of BLOCK_K
    # Each iteration loads a BK chunk from A and WT and accumulates into acc.
    for kk in range(0, D, BLOCK_K):
        k_offsets = kk + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < D

        # A indices: row = b*M + m_offsets, col = k_offsets
        # Pointer arithmetic: A is [B_total*M, D], row-major
        a_ptrs = A_ptr + (b * M + m_offsets[:, None]) * D + k_offsets[None, :]
        a_tile = tl.load(a_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0)

        # WT indices: [D, D], row = k_offsets, col = n_offsets
        wt_ptrs = WT_ptr + k_offsets[:, None] * D + n_offsets[None, :]
        wt_tile = tl.load(wt_ptrs, mask=(mask_k[:, None] & mask_n[None, :]), other=0.0)

        # Accumulate: (BLOCK_M x BLOCK_K) @ (BLOCK_K x BLOCK_N) -> (BLOCK_M x BLOCK_N)
        acc += tl.dot(a_tile, wt_tile)

    # Store results to Out[b, m, n]
    out_ptrs = Out_ptr + b * (M * D) + (m_offsets[:, None] * D) + n_offsets[None, :]
    store_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,      # [B, I, D]
        encoder_hidden_states: torch.Tensor,  # [B, T, D]
        process_weight: torch.Tensor,     # [D, D]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        - Do not use torch.cat or torch.matmul in forward.
        - Compute outputs via a Triton batched GEMM kernel twice (for encoder stream and hidden stream).
        Returns (processed_encoder_hidden_states: [B, T, D], processed_hidden_states: [B, I, D]).
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "All tensors must be on CUDA device for Triton kernels."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert process_weight.shape == (D, D), "process_weight must be [hidden_dim, hidden_dim]"
        # Ensure dtype is float32 for robust accumulation
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, \
            "This Triton implementation currently expects float32 tensors."

        # Allocate outputs
        processed_encoder = torch.empty((B, T, D), device=hidden_states.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, D), device=hidden_states.device, dtype=torch.float32)

        # Launch Triton kernel for encoder stream: M = T
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid_enc = (B, triton.cdiv(T, BLOCK_M), triton.cdiv(D, BLOCK_N))

        # Flatten encoder input to [B*T, D]
        A_encoder = encoder_hidden_states.view(-1, D).contiguous()  # shape [B*T, D]
        WT = process_weight.t().contiguous()  # shape [D, D]
        Out_encoder = processed_encoder.view(B, T, D)  # just a view, but ensure contiguous with empty above

        _batched_gemm_block_kernel[grid_enc](
            A_encoder, WT, Out_encoder,
            B_total=B,  # pid_b maps to batch b
            M=T, D=D,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Launch Triton kernel for hidden stream: M = I
        grid_hid = (B, triton.cdiv(I, BLOCK_M), triton.cdiv(D, BLOCK_N))
        A_hidden = hidden_states.view(-1, D).contiguous()  # shape [B*I, D]
        Out_hidden = processed_hidden.view(B, I, D)

        _batched_gemm_block_kernel[grid_hid](
            A_hidden, WT, Out_hidden,
            B_total=B,  # pid_b maps to batch b
            M=I, D=D,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
