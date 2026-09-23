import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_encoder_reduce_kernel(
    A_ptr,  # *float32, [B, T, D]
    W_ptr,  # *float32, [D, D]
    Out_ptr,  # *float32, [B, T, D]
    B: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
    stride_A_b, stride_A_t, stride_A_d,
    stride_W_k, stride_W_d,
    stride_Out_b, stride_Out_t, stride_Out_d,
    BM: tl.constexpr, BN: tl.constexpr, K_BLOCK: tl.constexpr,
):
    # program ids: batch, tiles over T, tiles over D
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BM + tl.arange(0, BM)  # rows along sequence (T)
    n_offsets = pid_n * BN + tl.arange(0, BN)  # columns along hidden (D)
    m_mask = m_offsets < T
    n_mask = n_offsets < D

    mm = m_offsets[:, None]   # [BM, 1]
    nn = n_offsets[None, :]   # [1, BN]
    mask_out = m_mask[:, None] & n_mask[None, :]  # [BM, BN]

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    # reduce over K in chunks
    for k0 in range(0, D, K_BLOCK):
        k_offsets = k0 + tl.arange(0, K_BLOCK)  # [K_BLOCK]
        k_mask = k_offsets < D

        # A[b, m, k] -> [BM, K_BLOCK]
        A_tile = tl.load(
            A_ptr + b * stride_A_b + mm * stride_A_t + k_offsets[None, :] * stride_A_d,
            mask=(m_mask[:, None] & k_mask[None, :]),
            other=0.0
        )
        # W[k, n] -> [K_BLOCK, BN]
        W_tile = tl.load(
            W_ptr + k_offsets[:, None] * stride_W_k + nn * stride_W_d,
            mask=(k_mask[:, None] & n_mask[None, :]),
            other=0.0
        )
        # outer-product accumulate: [BM, K_BLOCK] @ [K_BLOCK, BN] -> [BM, BN]
        acc += tl.dot(A_tile, W_tile)

    # store out[b, m, n] = acc[m, n]
    tl.store(Out_ptr + b * stride_Out_b + mm * stride_Out_t + nn * stride_Out_d, acc, mask=mask_out)


@triton.jit
def _gemm_hidden_reduce_kernel(
    A_ptr,  # *float32, [B, I, D]
    W_ptr,  # *float32, [D, D]
    Out_ptr,  # *float32, [B, I, D]
    B: tl.constexpr, I: tl.constexpr, D: tl.constexpr,
    stride_A_b, stride_A_i, stride_A_d,
    stride_W_k, stride_W_d,
    stride_Out_b, stride_Out_i, stride_Out_d,
    BM: tl.constexpr, BN: tl.constexpr, K_BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BM + tl.arange(0, BM)  # rows along I
    n_offsets = pid_n * BN + tl.arange(0, BN)  # columns along D
    m_mask = m_offsets < I
    n_mask = n_offsets < D

    mm = m_offsets[:, None]
    nn = n_offsets[None, :]
    mask_out = m_mask[:, None] & n_mask[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k0 in range(0, D, K_BLOCK):
        k_offsets = k0 + tl.arange(0, K_BLOCK)
        k_mask = k_offsets < D

        A_tile = tl.load(
            A_ptr + b * stride_A_b + mm * stride_A_i + k_offsets[None, :] * stride_A_d,
            mask=(m_mask[:, None] & k_mask[None, :]),
            other=0.0
        )
        W_tile = tl.load(
            W_ptr + k_offsets[:, None] * stride_W_k + nn * stride_W_d,
            mask=(k_mask[:, None] & n_mask[None, :]),
            other=0.0
        )
        acc += tl.dot(A_tile, W_tile)

    tl.store(Out_ptr + b * stride_Out_b + mm * stride_Out_i + nn * stride_Out_d, acc, mask=mask_out)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,         # [B, I, D]
        encoder_hidden_states: torch.Tensor, # [B, T, D]
        process_weight: torch.Tensor,        # [D, D]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Triton requires CUDA tensors and float32 for this implementation
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors for Triton."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "This Triton implementation expects float32 tensors."

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]

        # Ensure contiguity for predictable strides
        enc = encoder_hidden_states.contiguous()          # [B, T, D]
        hst = hidden_states.contiguous()                  # [B, I, D]
        WT = process_weight.t().contiguous()              # [D, D]

        # Allocate outputs
        processed_encoder = torch.empty((B, T, D), dtype=enc.dtype, device=enc.device)
        processed_hidden = torch.empty((B, I, D), dtype=hst.dtype, device=hst.device)

        # Choose moderate block sizes; loop over K reduces number of program instances
        BM = 32
        BN = 32
        K_BLOCK = 64  # reduce over K in 64-wide chunks

        # Grid over (batch, tiles of M, tiles of N)
        grid_enc = (B, triton.cdiv(T, BM), triton.cdiv(D, BN))
        _gemm_encoder_reduce_kernel[grid_enc](
            enc, WT, processed_encoder,
            B, T, D,
            enc.stride(0), enc.stride(1), enc.stride(2),
            WT.stride(0), WT.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BM=BM, BN=BN, K_BLOCK=K_BLOCK,
            num_warps=4, num_stages=2,
        )

        grid_hid = (B, triton.cdiv(I, BM), triton.cdiv(D, BN))
        _gemm_hidden_reduce_kernel[grid_hid](
            hst, WT, processed_hidden,
            B, I, D,
            hst.stride(0), hst.stride(1), hst.stride(2),
            WT.stride(0), WT.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BM=BM, BN=BN, K_BLOCK=K_BLOCK,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
