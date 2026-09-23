import torch
import triton
import triton.language as tl


@triton.jit
def _triton_gemm_right_matmul_kernel(
    A_ptr,        # pointer to A [M, K], contiguous or strided
    Wt_ptr,       # pointer to W^T [K, K] (process_weight transposed)
    C_ptr,        # pointer to C [M, K]
    M, K,         # int32: M = B*(T+I), K = H
    stride_am, stride_ak,  # strides for A
    stride_wtk, stride_wtn,  # strides for W^T
    stride_cm, stride_cn,  # strides for C
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D launch grid: (pid_m over tiles of M, pid_n over tiles of K)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    m_mask = m_offsets < M
    n_mask = n_offsets < K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = k_offsets < K

        # Load A tile [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Load W^T tile [BLOCK_K, BLOCK_N]
        w_ptrs = Wt_ptr + k_offsets[:, None] * stride_wtk + n_offsets[None, :] * stride_wtn
        w = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        # Accumulate
        acc += tl.dot(a, w)

    # Store C tile
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def _triton_run_gemm(out_cat: torch.Tensor, process_weight: torch.Tensor) -> torch.Tensor:
    """
    out_cat: [B, T+I, H], float32 on CUDA
    process_weight: [H, H], float32 on CUDA (no bias)
    Returns C: [B, T+I, H] = out_cat @ process_weight.T
    """
    assert out_cat.is_cuda and process_weight.is_cuda, "Tensors must be CUDA"
    B, L, H = out_cat.shape
    M = B * L
    # Ensure dtypes
    A = out_cat
    # W^T: [H, H]
    Wt = process_weight.t()  # [H, H]

    # Allocate output C [M, H]
    C = torch.empty((M, H), dtype=torch.float32, device=out_cat.device)

    # Choose tiling
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))

    _triton_gemm_right_matmul_kernel[grid](
        A, Wt, C,
        M, H,
        A.stride(0), A.stride(1),        # stride_am, stride_ak
        Wt.stride(0), Wt.stride(1),      # stride_wtk, stride_wtn
        C.stride(0), C.stride(1),        # stride_cm, stride_cn
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )

    # Reshape back to [B, L, H]
    return C.view(B, L, H)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension
        - Apply linear projection via Triton GEMM (right-multiply by process_weight.T)
        - Split back into separate encoder and image streams

        Returns:
            processed_encoder: [B, T, H]
            processed_hidden: [B, I, H]
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA for Triton"
        # Step 1: Concatenate along sequence dimension in torch (robust and simple)
        # Shapes: hidden_states [B, I, H], encoder_hidden_states [B, T, H]
        out_cat = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [B, T+I, H]

        # Step 2: Triton GEMM: C = out_cat @ process_weight.T
        processed = _triton_run_gemm(out_cat, process_weight)  # [B, T+I, H]

        # Step 3: Split into encoder and hidden streams
        B, L, H = processed.shape
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
