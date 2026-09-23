import torch
import triton
import triton.language as tl


# Triton kernel: concatenate along sequence dimension into out_cat [B, L, H]
# out_cat[b, m, k] = encoder[b, m, k] if m < T else hidden[b, m - T, k]
@triton.jit
def concat_seq_kernel(
    out_ptr, enc_ptr, hid_ptr,
    B, T: tl.constexpr, I: tl.constexpr, H: tl.constexpr,
    stride_ob, stride_ol, stride_oh,
    stride_eb, stride_et, stride_eh,
    stride_hb, stride_hi, stride_hh,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D launch: (batch, tiles along L, tiles along H)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)

    L = T + I

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # along sequence (L)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)  # along hidden dim (H)

    mask_m = offs_m < L
    mask_k = offs_k < H

    # Output pointers for this tile
    out_ptrs = out_ptr + pid_b * stride_ob + offs_m[:, None] * stride_ol + offs_k[None, :] * stride_oh

    # Masks for regions
    mask_encoder = mask_m[:, None] & mask_k[None, :]
    mask_hidden = (offs_m[:, None] >= T) & (offs_m[:, None] < L) & mask_k[None, :]

    # Load encoder values for first T rows
    enc_ptrs = enc_ptr + pid_b * stride_eb + offs_m[:, None] * stride_et + offs_k[None, :] * stride_eh
    enc_vals = tl.load(enc_ptrs, mask=mask_encoder, other=0.0)

    # Load hidden values for remaining I rows
    hid_ptrs = hid_ptr + pid_b * stride_hb + (offs_m[:, None] - T) * stride_hi + offs_k[None, :] * stride_hh
    hid_vals = tl.load(hid_ptrs, mask=mask_hidden, other=0.0)

    # Store into output
    tl.store(out_ptrs, enc_vals, mask=mask_encoder)
    tl.store(out_ptrs, hid_vals, mask=mask_hidden)


# Triton kernel: batched GEMM on A [B, M=L, K=H] and B [K=H, N=H] -> C [B, M=L, N=H]
# We accumulate in fp32 for stability; inputs/weights are assumed fp32 in this module.
@triton.jit
def batched_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    BATCH: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_ab, stride_am, stride_ak,   # A: [B, M, K]
    stride_bb, stride_bk, stride_bn,   # B: [K, N] (W_T)
    stride_cb, stride_cm, stride_cn,   # C: [B, M, N]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)  # batch
    pid_m = tl.program_id(1)  # tiles along M (sequence)
    pid_n = tl.program_id(2)  # tiles along N (hidden)

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in blocks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A and B
        a_ptrs = A_ptr + pid_b * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        # Masks for bounds
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load and multiply-accumulate
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)

    # Write back
    c_ptrs = C_ptr + pid_b * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension.
        - Apply linear projection using process_weight.T via Triton GEMM.
        - Split back into encoder and hidden streams.
        """
        # Ensure inputs are on the same device and contiguous
        device = hidden_states.device
        # Cast to float32 for numerical stability and to simplify Triton kernel assumptions
        enc = encoder_hidden_states.contiguous().to(torch.float32)
        hid = hidden_states.contiguous().to(torch.float32)
        W = process_weight.t().contiguous().to(torch.float32)  # [H, H]

        B, T, H = enc.shape
        _, I, _ = hid.shape
        assert I == H, "hidden_states last dim must equal hidden_dim"
        assert W.shape == (H, H), "process_weight must be [hidden_dim, hidden_dim]"

        L = T + I

        # 1) Concatenate into out_cat [B, L, H] using Triton
        out_cat = torch.empty((B, L, H), device=device, dtype=torch.float32)

        # Choose fixed tiles to ensure grid covers all (L, H) without autotune issues
        BLOCK_M = 64  # tiles along sequence dimension (L)
        BLOCK_K = 64  # tiles along hidden dimension (H)

        grid_concat = (B, triton.cdiv(L, BLOCK_M), triton.cdiv(H, BLOCK_K))

        # Strides
        stride_ob = out_cat.stride(0)
        stride_ol = out_cat.stride(1)
        stride_oh = out_cat.stride(2)

        stride_eb = enc.stride(0)
        stride_et = enc.stride(1)
        stride_eh = enc.stride(2)

        stride_hb = hid.stride(0)
        stride_hi = hid.stride(1)
        stride_hh = hid.stride(2)

        concat_seq_kernel[grid_concat](
            out_cat, enc, hid,
            B, T, I, H,
            stride_ob, stride_ol, stride_oh,
            stride_eb, stride_et, stride_eh,
            stride_hb, stride_hi, stride_hh,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # 2) Batched GEMM: out_cat [B, L, H] @ W_T [H, H] -> processed [B, L, H]
        processed = torch.empty((B, L, H), device=device, dtype=torch.float32)

        # Strides for GEMM
        stride_ab = out_cat.stride(0)  # A: [B, M=L, K=H]
        stride_am = out_cat.stride(1)
        stride_ak = out_cat.stride(2)

        stride_bb = W.stride(0)  # B: [K=H, N=H]
        stride_bk = W.stride(1)
        stride_bn = W.stride(2)

        stride_cb = processed.stride(0)  # C: [B, M=L, N=H]
        stride_cm = processed.stride(1)
        stride_cn = processed.stride(2)

        # Fixed tiles for GEMM; these are friendly for H=128 and moderate L
        BLOCK_M2 = 64
        BLOCK_N2 = 64
        BLOCK_K2 = 64

        grid_gemm = (B, triton.cdiv(L, BLOCK_M2), triton.cdiv(H, BLOCK_N2))

        batched_gemm_kernel[grid_gemm](
            out_cat, W, processed,
            BATCH=B, M=L, N=H, K=H,
            stride_ab=stride_ab, stride_am=stride_am, stride_ak=stride_ak,
            stride_bb=stride_bb, stride_bk=stride_bk, stride_bn=stride_bn,
            stride_cb=stride_cb, stride_cm=stride_cm, stride_cn=stride_cn,
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2,
            num_warps=4, num_stages=3,
        )

        # 3) Split back into encoder and hidden streams
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]

        # Return as float32 (original run used no dtype conversion; here we keep fp32 for stability)
        return processed_encoder, processed_hidden


# For completeness, a helper to mimic the original run signature
@torch.no_grad()
def run(hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return ModelNew()(hidden_states, encoder_hidden_states, process_weight)


# Original Model can use the same forward; ensure tensors are on CUDA for Triton
# class Model(torch.nn.Module):
#     def forward(self, *args):
#         return run(*args)


def run(*args):
    return ModelNew()(*args)
