import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def batched_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    BATCH: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D launch: (batch, tiles along M, tiles along N)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for the first K-block
    a_ptrs = A_ptr + pid_b * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_mask = (k + offs_k) < K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        # a: [BLOCK_M, BLOCK_K], b: [BLOCK_K, BLOCK_N] -> tl.dot returns [BLOCK_M, BLOCK_N]
        acc += tl.dot(a, b)
        # advance pointers by BLOCK_K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Store results
    c_ptrs = C_ptr + pid_b * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N'],
)
@triton.jit
def concat_seq_kernel(
    e_ptr, h_ptr, out_ptr,
    BATCH: tl.constexpr, T: tl.constexpr, I: tl.constexpr, H: tl.constexpr,
    stride_eb, stride_et, stride_ek,
    stride_hb, stride_hi, stride_hk,
    stride_ob, stride_ot, stride_ok,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Grid over (batch, sequence positions in tiles, hidden dim in tiles)
    pid_b = tl.program_id(0)
    pid_pos = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_pos = pid_pos * BLOCK_M + tl.arange(0, BLOCK_M)  # along sequence L = T + I
    offs_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)      # along hidden dim H

    pos_mask = offs_pos < (T + I)
    k_mask = offs_k < H

    # Determine if this position comes from encoder (first T) or hidden (next I)
    is_encoder = offs_pos[:, None] < T  # [BLOCK_M, 1], broadcast across k

    # Pointers
    e_ptrs = e_ptr + pid_b * stride_eb + offs_pos[:, None] * stride_et + offs_k[None, :] * stride_ek
    h_ptrs = h_ptr + pid_b * stride_hb + (offs_pos[:, None] - T) * stride_hi + offs_k[None, :] * stride_hk

    val_e = tl.load(e_ptrs, mask=pos_mask[:, None] & k_mask[None, :] & is_encoder, other=0.0)
    val_h = tl.load(h_ptrs, mask=pos_mask[:, None] & k_mask[None, :] & (~is_encoder), other=0.0)
    val = val_e + val_h

    out_ptrs = out_ptr + pid_b * stride_ob + offs_pos[:, None] * stride_ot + offs_k[None, :] * stride_ok
    tl.store(out_ptrs, val, mask=pos_mask[:, None] & k_mask[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized implementation:
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension using Triton.
        - Applies linear projection using Triton batched matmul (no torch.matmul).
        - Splits the result back into encoder and hidden streams.
        """
        assert hidden_states.ndim == 3 and encoder_hidden_states.ndim == 3
        assert process_weight.ndim == 2
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H
        assert process_weight.shape[0] == H and process_weight.shape[1] == H

        device = hidden_states.device
        dtype = hidden_states.dtype

        # Ensure contiguity for Triton kernels
        e = encoder_hidden_states.contiguous()
        h = hidden_states.contiguous()
        W = process_weight.contiguous()

        # 1) Concatenate along sequence dimension using Triton
        L = T + I
        out_cat = torch.empty((B, L, H), dtype=dtype, device=device)

        grid_concat = lambda META: (
            B,
            triton.cdiv(L, META['BLOCK_M']),
            triton.cdiv(H, META['BLOCK_N']),
        )
        concat_seq_kernel[grid_concat](
            e, h, out_cat,
            BATCH=B, T=T, I=I, H=H,
            stride_eb=e.stride(0), stride_et=e.stride(1), stride_ek=e.stride(2),
            stride_hb=h.stride(0), stride_hi=h.stride(1), stride_hk=h.stride(2),
            stride_ob=out_cat.stride(0), stride_ot=out_cat.stride(1), stride_ok=out_cat.stride(2),
        )

        # 2) Linear projection: C = out_cat @ W.T, W_T shape [H, H]
        W_T = W.t()  # Triton will expect contiguous strides, .t().contiguous() is fine

        # Output tensor in fp32 for numerical stability
        C = torch.empty((B, L, H), dtype=torch.float32, device=device)

        grid_gemm = lambda META: (
            B,
            triton.cdiv(L, META['BLOCK_M']),
            triton.cdiv(H, META['BLOCK_N']),
        )
        batched_matmul_kernel[grid_gemm](
            out_cat, W_T, C,
            BATCH=B, M=L, N=H, K=H,
            stride_ab=out_cat.stride(0), stride_am=out_cat.stride(1), stride_ak=out_cat.stride(2),
            stride_bb=W_T.stride(0), stride_bk=W_T.stride(1), stride_bn=W_T.stride(2),
            stride_cb=C.stride(0), stride_cm=C.stride(1), stride_cn=C.stride(2),
        )

        # Cast back to original dtype
        C = C.to(dtype)

        # 3) Split back into encoder and hidden streams
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
