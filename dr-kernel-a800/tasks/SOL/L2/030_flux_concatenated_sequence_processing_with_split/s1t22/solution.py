import torch
import triton
import triton.language as tl


# Triton kernel: concatenate [B, T, K] and [B, P, K] into [B, T+P, K]
# Grid: (B, T+P)
@triton.jit
def _concat_seq_kernel(
    e_ptr, h_ptr, out_ptr,
    B, T, P, K,
    e_b_stride, e_t_stride, e_k_stride,
    h_b_stride, h_p_stride, h_k_stride,
    o_b_stride, o_l_stride, o_k_stride,
    BLOCK_K: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    l = pid_l
    if l >= (T + P):
        return

    k_offsets = tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    if l < T:
        base_e = e_ptr + pid_b * e_b_stride + l * e_t_stride
        out_base = out_ptr + pid_b * o_b_stride + l * o_l_stride
        vals = tl.load(base_e + k_offsets * e_k_stride, mask=mask_k, other=0.0)
        tl.store(out_base + k_offsets * o_k_stride, vals, mask=mask_k)
    else:
        base_h = h_ptr + pid_b * h_b_stride + (l - T) * h_p_stride
        out_base = out_ptr + pid_b * o_b_stride + l * o_l_stride
        vals = tl.load(base_h + k_offsets * h_k_stride, mask=mask_k, other=0.0)
        tl.store(out_base + k_offsets * o_k_stride, vals, mask=mask_k)


# Triton kernel: split [B, T+P, K] into [B, T, K] (encoder stream)
# Grid: (B, T)
@triton.jit
def _split_encoder_kernel(
    in_ptr, out_ptr,
    B, T, P, K,
    in_b_stride, in_l_stride, in_k_stride,
    out_b_stride, out_l_stride, out_k_stride,
    BLOCK_K: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    i = pid_i
    if i >= T:
        return

    k_offsets = tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    in_base = in_ptr + pid_b * in_b_stride + i * in_l_stride
    out_base = out_ptr + pid_b * out_b_stride + i * out_l_stride

    vals = tl.load(in_base + k_offsets * in_k_stride, mask=mask_k, other=0.0)
    tl.store(out_base + k_offsets * out_k_stride, vals, mask=mask_k)


# Triton kernel: split [B, T+P, K] into [B, P, K] (hidden stream)
# Grid: (B, P)
@triton.jit
def _split_hidden_kernel(
    in_ptr, out_ptr,
    B, T, P, K,
    in_b_stride, in_l_stride, in_k_stride,
    out_b_stride, out_p_stride, out_k_stride,
    BLOCK_K: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_j = tl.program_id(1)
    j = pid_j
    if j >= P:
        return

    k_offsets = tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    in_base = in_ptr + pid_b * in_b_stride + (j + T) * in_l_stride
    out_base = out_ptr + pid_b * out_b_stride + j * out_p_stride

    vals = tl.load(in_base + k_offsets * in_k_stride, mask=mask_k, other=0.0)
    tl.store(out_base + k_offsets * out_k_stride, vals, mask=mask_k)


# Triton kernel: GEMM for [B, T+P, K] @ [K, K] -> [B, T+P, K]
# 3D grid: (B, tiles over S, tiles over K)
@triton.jit
def _gemm_bsp_kernel(
    A_ptr, W_ptr, C_ptr,
    B, S, K,  # S = T + P
    A_b_stride, A_l_stride, A_k_stride,
    W_k_stride, W_n_stride,  # W is [K, K]
    C_b_stride, C_l_stride, C_k_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    mask_m = m_offsets < S
    mask_n = n_offsets < K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = k_offsets < K

        # A_tile: [BLOCK_M, BLOCK_K]
        A_tile = tl.load(
            A_ptr + pid_b * A_b_stride + m_offsets[:, None] * A_l_stride + k_offsets[None, :] * A_k_stride,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0
        )

        # W_tile: [BLOCK_K, BLOCK_N]
        W_tile = tl.load(
            W_ptr + k_offsets[:, None] * W_k_stride + n_offsets[None, :] * W_n_stride,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0
        )

        acc += tl.dot(A_tile, W_tile)

    # Store tile C[b, m, n]
    tl.store(
        C_ptr + pid_b * C_b_stride + m_offsets[:, None] * C_l_stride + n_offsets[None, :] * C_k_stride,
        acc,
        mask=mask_m[:, None] & mask_n[None, :]
    )


class ModelNew(torch.nn.Module):
    def __init__(self, block_k: int = 64, block_m: int = 128, block_n: int = 128):
        super().__init__()
        self.block_k = block_k
        self.block_m = block_m
        self.block_n = block_n

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [B, P, K]
        encoder_hidden_states: [B, T, K]
        process_weight: [K, K]
        Returns: (processed_encoder: [B, T, K], processed_hidden: [B, P, K])
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be [B, *, K]"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        P = hidden_states.shape[1]
        K = hidden_states.shape[2]
        assert process_weight.shape == (K, K), "process_weight must be [K, K]"

        # Concatenate into [B, T+P, K]
        S = T + P
        concatenated = torch.empty((B, S, K), device=hidden_states.device, dtype=hidden_states.dtype)

        BLOCK_K = min(self.block_k, K)
        grid_concat = (B, S)
        _concat_seq_kernel[grid_concat](
            encoder_hidden_states, hidden_states, concatenated,
            B, T, P, K,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # GEMM: [B, S, K] @ [K, K] -> [B, S, K]
        C = torch.empty((B, S, K), device=hidden_states.device, dtype=hidden_states.dtype)

        grid_gemm = (B, triton.cdiv(S, self.block_m), triton.cdiv(K, self.block_n))
        _gemm_bsp_kernel[grid_gemm](
            concatenated, process_weight, C,
            B, S, K,
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            process_weight.stride(0), process_weight.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=self.block_m, BLOCK_N=self.block_n, BLOCK_K=self.block_k,
            num_warps=4, num_stages=2
        )

        # Split C into encoder and hidden outputs
        processed_encoder = torch.empty((B, T, K), device=hidden_states.device, dtype=hidden_states.dtype)
        processed_hidden = torch.empty((B, P, K), device=hidden_states.device, dtype=hidden_states.dtype)

        grid_split_e = (B, T)
        _split_encoder_kernel[grid_split_e](
            C, processed_encoder,
            B, T, P, K,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        grid_split_h = (B, P)
        _split_hidden_kernel[grid_split_h](
            C, processed_hidden,
            B, T, P, K,
            C.stride(0), C.stride(1), C.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
