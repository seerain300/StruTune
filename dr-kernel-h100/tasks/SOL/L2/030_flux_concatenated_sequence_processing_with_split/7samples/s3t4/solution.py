import torch
import triton
import triton.language as tl

@triton.jit
def _concat_sequences_kernel(
    out_ptr,            # *fp32, [N, L_total, K]
    encoder_ptr,        # *fp32, [N, L_txt, K]
    hidden_ptr,         # *fp32, [N, L_img, K]
    N, L_txt, L_img, K,
    stride_oe_n, stride_oe_t, stride_oe_k,  # strides for encoder output
    stride_hs_n, stride_hs_t, stride_hs_k,  # strides for hidden input
    stride_out_n, stride_out_t, stride_out_k,  # strides for out
    BLOCK_K: tl.constexpr,
):
    # 3D grid: (N, L_total, tiles along K)
    n = tl.program_id(0)
    t = tl.program_id(1)  # 0 .. L_total-1
    k_tile = tl.program_id(2)
    k_offsets = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # Select source based on t
    is_encoder = t < L_txt
    # Load vector from source
    # Use tl.where to select pointer/strides; but better to branch by scalar is_encoder
    if is_encoder:
        base = encoder_ptr + n * stride_oe_n + t * stride_oe_t
    else:
        base = hidden_ptr + n * stride_hs_n + (t - L_txt) * stride_hs_t

    # Load values across K tile
    vals = tl.load(base + k_offsets * stride_hs_k, mask=mask_k, other=0.0)

    # Store to output at [n, t, k_offsets]
    out_base = out_ptr + n * stride_out_n + t * stride_out_t
    tl.store(out_base + k_offsets * stride_out_k, vals, mask=mask_k)


@triton.jit
def _batched_matmul_kernel(
    C,          # *fp32, [N_rows, K], will hold output per row
    A,          # *fp32, [N_rows, K], rows come from concatenated tensor
    B,          # *fp32, [K, K], process_weight.T
    N_rows, K,
    BLOCK_M: tl.constexpr,  # number of rows per program
    BLOCK_N: tl.constexpr,  # output columns per tile
    BLOCK_K: tl.constexpr,  # reduction chunk
):
    # Each program handles BLOCK_M rows starting at pid_m
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)  # tile along output columns
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # row indices
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # output column indices

    # Masks for rows/cols
    mask_m = m_offsets < N_rows
    mask_n = n_offsets < K

    # Accumulator for [BLOCK_M, BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction loop over K
    for k0 in tl.static_range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # reduction indices
        mask_k = k_offsets < K

        # Load A_rows[m, k] -> shape [BLOCK_M, BLOCK_K]
        a_ptrs = A + m_offsets[:, None] * K + k_offsets[None, :]  # A is [N_rows, K] contiguous
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load B[k, n] -> shape [BLOCK_K, BLOCK_N]
        b_ptrs = B + k_offsets[:, None] * K + n_offsets[None, :]  # B is [K, K] contiguous
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # acc += a @ b
        acc += tl.dot(a, b)

    # Store results to C[m, n] for valid rows
    c_ptrs = C + m_offsets[:, None] * K + n_offsets[None, :]  # C is [N_rows, K] contiguous
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension using Triton.
        - Compute processed = concatenated @ process_weight.T using a tiled Triton GEMM.
        - Split outputs back into two streams.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "This implementation expects float32 tensors."

        N = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        K = hidden_states.shape[2]
        L_total = L_txt + L_img

        # 1) Concatenate along sequence dimension using Triton
        out_cat = torch.empty((N, L_total, K), device=hidden_states.device, dtype=torch.float32)

        # Compute strides for input tensors (assumed contiguous [N, L, K])
        stride_oe_n, stride_oe_t, stride_oe_k = encoder_hidden_states.stride()
        stride_hs_n, stride_hs_t, stride_hs_k = hidden_states.stride()
        stride_out_n, stride_out_t, stride_out_k = out_cat.stride()

        BLOCK_K = 128  # tile size along K for concatenation
        grid_concat = (N, L_total, triton.cdiv(K, BLOCK_K))
        _concat_sequences_kernel[grid_concat](
            out_cat, encoder_hidden_states, hidden_states,
            N, L_txt, L_img, K,
            stride_oe_n, stride_oe_t, stride_oe_k,
            stride_hs_n, stride_hs_t, stride_hs_k,
            stride_out_n, stride_out_t, stride_out_k,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=2,
        )

        # 2) GEMM: A_rows = out_cat reshaped to [N_rows, K], B = process_weight.T (shape [K, K])
        # Flatten rows across batch and sequence
        A_rows = out_cat.view(-1, K)  # shape [N_rows, K], N_rows = N * L_total
        N_rows = A_rows.shape[0]
        B = process_weight.t().contiguous()  # [K, K]

        # Allocate output for rows
        C_rows = torch.empty((N_rows, K), device=hidden_states.device, dtype=torch.float32)

        # Tiling parameters for GEMM
        # Choose BLOCK_M, BLOCK_N, BLOCK_K based on K for better performance
        # Heuristic:
        if K >= 4096:
            BLOCK_M = 8
            BLOCK_N = 256
            BLOCK_K = 128
            num_warps = 8
        elif K >= 2048:
            BLOCK_M = 16
            BLOCK_N = 256
            BLOCK_K = 128
            num_warps = 8
        elif K >= 1024:
            BLOCK_M = 16
            BLOCK_N = 128
            BLOCK_K = 128
            num_warps = 4
        else:
            BLOCK_M = 16
            BLOCK_N = 64
            BLOCK_K = 64
            num_warps = 4

        grid_gemm = (triton.cdiv(N_rows, BLOCK_M), triton.cdiv(K, BLOCK_N))
        _batched_matmul_kernel[grid_gemm](
            C_rows, A_rows, B, N_rows, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=num_warps, num_stages=2,
        )

        # 3) Reshape back to [N, L_total, K] and split into two streams
        processed = C_rows.view(N, L_total, K)
        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
