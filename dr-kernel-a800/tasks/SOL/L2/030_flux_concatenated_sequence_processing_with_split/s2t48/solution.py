import torch
import triton
import triton.language as tl


# Triton kernel: concatenate encoder_hidden_states [B, T, H] and hidden_states [B, I, H] into
# out_cat [B, L, H], where L = T + I. We tile over (B, sequence L, H).
@triton.jit
def concat_seq_kernel(
    out_ptr, encoder_ptr, hidden_ptr,
    B, T, I, H,  # dynamic sizes
    stride_ob, stride_ol, stride_oh,
    stride_eb, stride_et, stride_eh,
    stride_hb, stride_hi, stride_hh,
    BLOCK_M: tl.constexpr,  # tile along sequence dimension (L)
    BLOCK_K: tl.constexpr,  # tile along hidden dimension (H)
):
    pid_b = tl.program_id(0)  # batch
    pid_m = tl.program_id(1)  # tile along sequence
    pid_k = tl.program_id(2)  # tile along hidden

    L = T + I

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # sequence indices in [0, L)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)  # hidden indices in [0, H)

    # Masks for bounds
    mask_m = offs_m < L
    mask_k = offs_k < H

    # Compute output pointers for this tile: out[b, m, k]
    out_ptrs = out_ptr + pid_b * stride_ob + offs_m[:, None] * stride_ol + offs_k[None, :] * stride_oh

    # Region masks: encoder vs hidden
    mask_encoder = mask_m[:, None] & mask_k[None, :] & (offs_m[:, None] < T)
    mask_hidden = mask_m[:, None] & mask_k[None, :] & (offs_m[:, None] >= T)

    # Load and store for encoder region
    enc_ptrs = encoder_ptr + pid_b * stride_eb + offs_m[:, None] * stride_et + offs_k[None, :] * stride_eh
    enc_vals = tl.load(enc_ptrs, mask=mask_encoder, other=0.0)
    tl.store(out_ptrs, enc_vals, mask=mask_encoder)

    # Load and store for hidden region
    hid_ptrs = hidden_ptr + pid_b * stride_hb + (offs_m[:, None] - T) * stride_hi + offs_k[None, :] * stride_hh
    hid_vals = tl.load(hid_ptrs, mask=mask_hidden, other=0.0)
    tl.store(out_ptrs, hid_vals, mask=mask_hidden)


# Triton kernel: batched GEMM on out_cat [B, M=L, K=H] and W_T [K=H, N=H] -> out [B, M=L, N=H]
# 3D grid: (batch, tiles along M, tiles along N), loop over K dimension.
@triton.jit
def batched_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    BATCH: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_ab, stride_am, stride_ak,   # A: [B, M, K]
    stride_bb, stride_bk, stride_bn,   # B: [K, N] (W_T)
    stride_cb, stride_cm, stride_cn,   # C: [B, M, N]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)  # tiles along M (sequence)
    pid_n = tl.program_id(2)  # tiles along N (hidden)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in tiles
    for k in range(0, K, BLOCK_K):
        a_ptrs = A_ptr + pid_b * stride_ab + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak
        b_ptrs = B_ptr + (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        # Accumulate in fp32
        acc += tl.dot(a, b)

    # Write results
    c_ptrs = C_ptr + pid_b * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function:
        1) Concatenate encoder_hidden_states and hidden_states along sequence dimension into out_cat [B, L, H].
        2) Compute processed = out_cat @ process_weight.T using a Triton batched GEMM.
        3) Split processed into encoder and hidden streams: [B, T, H] and [B, I, H].
        Returns (processed_encoder, processed_hidden).
        """
        # Ensure tensors are on CUDA and contiguous
        device = hidden_states.device
        assert device.type == 'cuda', "Triton requires CUDA tensors."
        B, T, H = encoder_hidden_states.shape
        _, I, H2 = hidden_states.shape
        assert H == H2, "hidden_dim mismatch between encoder_hidden_states and hidden_states"
        L = T + I

        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        # process_weight is [H, H], need W_T = [H, H] (no bias)
        W_T = process_weight.t().contiguous()  # [H, H]

        # Allocate concatenated tensor [B, L, H], compute in float32
        out_cat = torch.empty((B, L, H), device=device, dtype=torch.float32)

        # Launch Triton concat kernel: grid over (B, tiles along L, tiles along H)
        BLOCK_M = 64  # tile along sequence (L)
        BLOCK_K = 64  # tile along hidden (H)
        grid = (B, triton.cdiv(L, BLOCK_M), triton.cdiv(H, BLOCK_K))
        concat_seq_kernel[grid](
            out_cat, enc, hid,
            B, T, I, H,
            out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
            enc.stride(0), enc.stride(1), enc.stride(2),
            hid.stride(0), hid.stride(1), hid.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Allocate output processed [B, L, H]
        processed = torch.empty((B, L, H), device=device, dtype=torch.float32)

        # Batched GEMM: A = out_cat [B, M=L, K=H], B = W_T [K=H, N=H], C = processed [B, M, N]
        BLOCK_M_GEMM = 64
        BLOCK_N_GEMM = 64
        BLOCK_K_GEMM = 64
        grid_gemm = (B, triton.cdiv(L, BLOCK_M_GEMM), triton.cdiv(H, BLOCK_N_GEMM))
        batched_gemm_kernel[grid_gemm](
            out_cat, W_T, processed,
            BATCH=B, M=L, N=H, K=H,
            stride_ab=out_cat.stride(0), stride_am=out_cat.stride(1), stride_ak=out_cat.stride(2),
            stride_bb=W_T.stride(0), stride_bk=W_T.stride(1), stride_bn=W_T.stride(2),
            stride_cb=processed.stride(0), stride_cm=processed.stride(1), stride_cn=processed.stride(2),
            BLOCK_M=BLOCK_M_GEMM, BLOCK_N=BLOCK_N_GEMM, BLOCK_K=BLOCK_K_GEMM,
            num_warps=4, num_stages=2,
        )

        # Split streams: [B, T, H] and [B, I, H]
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]

        return processed_encoder, processed_hidden


# If you need a small self-test (optional), you can compare against PyTorch reference:
# def _reference(hidden_states, encoder_hidden_states, process_weight):
#     concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)
#     processed = torch.matmul(concatenated, process_weight.t())
#     processed_encoder, processed_hidden = processed[:, :encoder_hidden_states.shape[1], :], processed[:, encoder_hidden_states.shape[1]:, :]
#     return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
