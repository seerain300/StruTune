import torch
import triton
import triton.language as tl


@triton.jit
def concat_seq_dim1_kernel(
    out_ptr,        # *fp32, output A: [B, M, K], M = T + I
    x1_ptr,         # *fp32, encoder_hidden_states: [B, T, K]
    x2_ptr,         # *fp32, hidden_states: [B, I, K]
    B: tl.int32, T: tl.int32, I: tl.int32, K: tl.int32,
    stride_out_b: tl.int32, stride_out_m: tl.int32, stride_out_k: tl.int32,
    stride_x1_b: tl.int32, stride_x1_t: tl.int32, stride_x1_k: tl.int32,
    stride_x2_b: tl.int32, stride_x2_i: tl.int32, stride_x2_k: tl.int32,
):
    # 3D grid: (B, tiles along M, tiles along K)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Tile sizes
    BLOCK_M = 128
    BLOCK_K = 64

    # Offsets within the tile
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    M_total = T + I
    mask_m = m_offsets < M_total
    mask_k = k_offsets < K

    # Determine source tensor per m: if m < T, use x1; else use x2
    use_x1 = m_offsets < T

    # Compute source indices for b and m
    # Note: we need b index for both x1 and x2; pid_b selects batch
    b = pid_b

    # Load from x1 for m < T, else load from x2
    # For out-of-range m, load zeros
    x1_addr = x1_ptr + b * stride_x1_b + (m_offsets[:, None] * 0) * stride_x1_t + k_offsets[None, :] * stride_x1_k
    x2_addr = x2_ptr + b * stride_x2_b + (m_offsets[:, None] - T) * stride_x2_i + k_offsets[None, :] * stride_x2_k

    x1_mask = mask_m[:, None] & mask_k[None, :]
    x2_mask = mask_m[:, None] & mask_k[None, :]
    val_x1 = tl.load(x1_addr, mask=x1_mask, other=0.0)
    val_x2 = tl.load(x2_addr, mask=x2_mask, other=0.0)

    # Select source per row
    src = tl.where(use_x1[:, None], val_x1, val_x2)

    # Store to out
    out_addr = out_ptr + b * stride_out_b + m_offsets[:, None] * stride_out_m + k_offsets[None, :] * stride_out_k
    out_mask = mask_m[:, None] & mask_k[None, :]
    tl.store(out_addr, src, mask=out_mask)


@triton.jit
def batched_matmul_kernel(
    C_ptr,          # *fp32, output [B, M, K], N = K
    A_ptr,          # *fp32, input [B, M, K], A = concatenated sequences
    W_ptr,          # *fp32, weight [K, K] = process_weight.T
    B: tl.int32, M: tl.int32, K: tl.int32,
    stride_ab: tl.int32, stride_am: tl.int32, stride_ak: tl.int32,
    stride_wk: tl.int32, stride_wn: tl.int32,
    stride_cb: tl.int32, stride_cm: tl.int32, stride_cn: tl.int32,
):
    # 3D grid: (B, tiles along M, tiles along N=K)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = m_offsets < M
    mask_n = n_offsets < K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A tile: [BM, BK]
        a_addr = A_ptr + pid_b * stride_ab + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = mask_m[:, None] & mask_k[None, :]
        a_tile = tl.load(a_addr, mask=a_mask, other=0.0)  # [BM, BK]

        # Load W tile: [BK, BN]
        w_addr = W_ptr + k_offsets[:, None] * stride_wk + n_offsets[None, :] * stride_wn
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_tile = tl.load(w_addr, mask=w_mask, other=0.0)  # [BK, BN]

        # Accumulate: (BM, BK) @ (BK, BN) -> (BM, BN)
        acc += tl.dot(a_tile, w_tile)

    # Store result
    c_addr = C_ptr + pid_b * stride_cb + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_addr, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        K = hidden_states.shape[2]  # hidden_dim

        # Ensure contiguity and dtype for Triton
        x1 = encoder_hidden_states.contiguous().to(torch.float32)   # [B, T, K]
        x2 = hidden_states.contiguous().to(torch.float32)           # [B, I, K]

        # 1) Concatenate along sequence dimension using Triton
        M = T + I
        A = torch.empty((B, M, K), device=x1.device, dtype=torch.float32)

        grid_concat = (B, triton.cdiv(M, 128), triton.cdiv(K, 64))
        concat_seq_dim1_kernel[grid_concat](
            A, x1, x2,
            B, T, I, K,
            A.stride(0), A.stride(1), A.stride(2),
            x1.stride(0), x1.stride(1), x1.stride(2),
            x2.stride(0), x2.stride(1), x2.stride(2),
            num_warps=4, num_stages=2,
        )

        # 2) Compute C = A @ W, where W = process_weight.T
        W = process_weight.t().contiguous().to(torch.float32)       # [K, K]
        C = torch.empty((B, M, K), device=A.device, dtype=torch.float32)

        grid_gemm = (B, triton.cdiv(M, 64), triton.cdiv(K, 64))
        batched_matmul_kernel[grid_gemm](
            C, A, W,
            B, M, K,
            A.stride(0), A.stride(1), A.stride(2),
            W.stride(0), W.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            num_warps=4, num_stages=2,
        )

        # 3) Split back into separate streams
        processed_encoder = C[:, :T, :]                     # [B, T, K]
        processed_hidden = C[:, T:, :]                     # [B, I, K]

        # Cast back to original input dtypes to match the original API
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
