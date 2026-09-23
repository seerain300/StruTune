import torch
import triton
import triton.language as tl


@triton.jit
def _batched_gemm_block_kernel(
    A_ptr,  # *fp32, [B, M, D]
    B_ptr,  # *fp32, [D, D] (process_weight.T)
    Out_ptr,  # *fp32, [B, M, D]
    B, M, D,  # int32 sizes
    A_s0, A_s1, A_s2,   # strides for A: (B, M, D)
    B_s0, B_s1,         # strides for B: (D, D), B_s0 = 0, B_s1 = 1 typically
    Out_s0, Out_s1, Out_s2,  # strides for Out: (B, M, D)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program ids
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # tile coordinates
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # accumulator [BLOCK_M, BLOCK_N] in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K dimension in chunks
    for kk in range(0, D, BLOCK_K):
        k_offsets = kk + tl.arange(0, BLOCK_K)

        # pointers for A tile: shape [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + pid_b * A_s0 + m_offsets[:, None] * A_s1 + k_offsets[None, :] * A_s2
        # pointers for B tile: shape [BLOCK_K, BLOCK_N], B is [D, D] with strides (B_s0, B_s1)
        B_ptrs = B_ptr + k_offsets[:, None] * B_s0 + n_offsets[None, :] * B_s1

        # masks for boundaries
        m_mask = m_offsets[:, None] < M
        n_mask = n_offsets[None, :] < D
        k_mask = k_offsets[:, None] < D  # both dims are D so this is consistent

        # load tiles
        A_tile = tl.load(A_ptrs, mask=m_mask & k_mask, other=0.0)  # [BLOCK_M, BLOCK_K]
        B_tile = tl.load(B_ptrs, mask=k_mask & n_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # accumulate
        acc += tl.dot(A_tile, B_tile)  # fp32 accumulate

    # store results
    Out_ptrs = Out_ptr + pid_b * Out_s0 + m_offsets[:, None] * Out_s1 + n_offsets[None, :] * Out_s2
    out_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < D)
    tl.store(Out_ptrs, acc, mask=out_mask)


@triton.jit
def _gemm_encoder_blocked_kernel(
    enc_ptr,    # *fp32, [B, T, D]
    WT_ptr,     # *fp32, [D, D] (process_weight.T)
    out_ptr,    # *fp32, [B, T, D]
    B: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
    enc_s0, enc_s1, enc_s2,
    WT_s0, WT_s1,
    out_s0, out_s1, out_s2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # grid over (B, tiles of T, tiles of D)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # along T
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # along D

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kk in range(0, D, BLOCK_K):
        k_offsets = kk + tl.arange(0, BLOCK_K)
        # A is [B, T, D]; load tile [BLOCK_M, BLOCK_K]
        A_ptrs = enc_ptr + pid_b * enc_s0 + m_offsets[:, None] * enc_s1 + k_offsets[None, :] * enc_s2
        # B is [D, D]; load tile [BLOCK_K, BLOCK_N]
        B_ptrs = WT_ptr + k_offsets[:, None] * WT_s0 + n_offsets[None, :] * WT_s1

        m_mask = m_offsets[:, None] < T
        n_mask = n_offsets[None, :] < D
        k_mask = k_offsets[:, None] < D

        A_tile = tl.load(A_ptrs, mask=m_mask & k_mask, other=0.0)
        B_tile = tl.load(B_ptrs, mask=k_mask & n_mask, other=0.0)

        acc += tl.dot(A_tile, B_tile)

    Out_ptrs = out_ptr + pid_b * out_s0 + m_offsets[:, None] * out_s1 + n_offsets[None, :] * out_s2
    out_mask = (m_offsets[:, None] < T) & (n_offsets[None, :] < D)
    tl.store(Out_ptrs, acc, mask=out_mask)


@triton.jit
def _gemm_hidden_blocked_kernel(
    hst_ptr,    # *fp32, [B, I, D]
    WT_ptr,     # *fp32, [D, D]
    out_ptr,    # *fp32, [B, I, D]
    B: tl.constexpr, I: tl.constexpr, D: tl.constexpr,
    hst_s0, hst_s1, hst_s2,
    WT_s0, WT_s1,
    out_s0, out_s1, out_s2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # along I
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # along D

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kk in range(0, D, BLOCK_K):
        k_offsets = kk + tl.arange(0, BLOCK_K)
        # A is [B, I, D]; load tile [BLOCK_M, BLOCK_K]
        A_ptrs = hst_ptr + pid_b * hst_s0 + m_offsets[:, None] * hst_s1 + k_offsets[None, :] * hst_s2
        # B is [D, D]; load tile [BLOCK_K, BLOCK_N]
        B_ptrs = WT_ptr + k_offsets[:, None] * WT_s0 + n_offsets[None, :] * WT_s1

        m_mask = m_offsets[:, None] < I
        n_mask = n_offsets[None, :] < D
        k_mask = k_offsets[:, None] < D

        A_tile = tl.load(A_ptrs, mask=m_mask & k_mask, other=0.0)
        B_tile = tl.load(B_ptrs, mask=k_mask & n_mask, other=0.0)

        acc += tl.dot(A_tile, B_tile)

    Out_ptrs = out_ptr + pid_b * out_s0 + m_offsets[:, None] * out_s1 + n_offsets[None, :] * out_s2
    out_mask = (m_offsets[:, None] < I) & (n_offsets[None, :] < D)
    tl.store(Out_ptrs, acc, mask=out_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized forward:
        - No torch.cat, no torch.matmul
        - Computes two batched GEMMs directly in Triton:
            processed_encoder = encoder_hidden_states @ process_weight.T  -> [B, T, D]
            processed_hidden = hidden_states @ process_weight.T           -> [B, I, D]
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == D and process_weight.shape[0] == D and process_weight.shape[1] == D, "Dimension mismatch"

        # Ensure fp32 and contiguity for robust Triton kernel behavior
        enc = encoder_hidden_states.contiguous().to(torch.float32)
        hst = hidden_states.contiguous().to(torch.float32)
        WT = process_weight.contiguous().to(torch.float32)  # [D, D]

        processed_encoder = torch.empty((B, T, D), device=enc.device, dtype=enc.dtype)
        processed_hidden = torch.empty((B, I, D), device=hst.device, dtype=hst.dtype)

        # Choose block sizes; these work well for typical dimensions in the benchmark
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64

        # Launch GEMM for encoder stream: [B, T, D] @ [D, D] -> [B, T, D]
        grid_enc = (B, triton.cdiv(T, BLOCK_M), triton.cdiv(D, BLOCK_N))
        _gemm_encoder_blocked_kernel[grid_enc](
            enc, WT, processed_encoder,
            B, T, D,
            enc.stride(0), enc.stride(1), enc.stride(2),
            WT.stride(0), WT.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Launch GEMM for hidden stream: [B, I, D] @ [D, D] -> [B, I, D]
        grid_hid = (B, triton.cdiv(I, BLOCK_M), triton.cdiv(D, BLOCK_N))
        _gemm_hidden_blocked_kernel[grid_hid](
            hst, WT, processed_hidden,
            B, I, D,
            hst.stride(0), hst.stride(1), hst.stride(2),
            WT.stride(0), WT.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
