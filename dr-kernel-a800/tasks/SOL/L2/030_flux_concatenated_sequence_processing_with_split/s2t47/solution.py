import torch
import triton
import triton.language as tl

# --------------------------
# Triton kernel: concatenate encoder and hidden sequences along sequence dim into out_cat [B, L, H]
# out_cat[b, m, k] = encoder[b, m, k] if m < T else hidden[b, m - T, k]
# 2D tiling over (L, H) per batch
# --------------------------
@triton.jit
def concat_encoder_hidden_kernel(
    out_ptr,  # *float32 [B, L, H]
    encoder_ptr,  # *float32 [B, T, H]
    hidden_ptr,  # *float32 [B, I, H]
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, H: tl.constexpr,
    stride_ob, stride_ol, stride_oh,
    stride_eb, stride_et, stride_eh,
    stride_hb, stride_hi, stride_hh,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)  # tiles over L
    pid_k = tl.program_id(2)  # tiles over H

    L = T + I

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # indices along sequence dim
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)  # indices along hidden dim

    mask_m = offs_m < L
    mask_k = offs_k < H

    # Prepare output pointers for this tile
    out_ptrs = out_ptr + pid_b * stride_ob + offs_m[:, None] * stride_ol + offs_k[None, :] * stride_oh

    # Compute which rows come from encoder vs hidden
    mask_encoder = mask_m[:, None] & mask_k[None, :]
    mask_hidden = (offs_m[:, None] >= T) & (offs_m[:, None] < L) & mask_k[None, :]

    # Load encoder values for m < T
    enc_ptrs = encoder_ptr + pid_b * stride_eb + offs_m[:, None] * stride_et + offs_k[None, :] * stride_eh
    enc_vals = tl.load(enc_ptrs, mask=mask_encoder, other=0.0)

    # Load hidden values for m >= T
    hid_ptrs = hidden_ptr + pid_b * stride_hb + (offs_m[:, None] - T) * stride_hi + offs_k[None, :] * stride_hh
    hid_vals = tl.load(hid_ptrs, mask=mask_hidden, other=0.0)

    # Store results
    tl.store(out_ptrs, enc_vals, mask=mask_encoder)
    tl.store(out_ptrs, hid_vals, mask=mask_hidden)

# --------------------------
# Triton kernel: batched GEMM on out_cat [B, M=L, K=H] and W_T [K=H, N=H] -> out [B, M=L, N=H]
# 3D grid: (batch, tiles along M, tiles along N), loop over K
# --------------------------
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + pid_b * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile: [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store C tile
    c_ptrs = C_ptr + pid_b * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)

# --------------------------
# ModelNew: forward must invoke Triton kernels
# --------------------------
class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
          - Concatenate sequences in Triton into [B, L, H]
          - Apply linear projection via Triton GEMM
          - Split back into [B, T, H] and [B, I, H]
        """
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        L = T + I

        # Ensure inputs are contiguous
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        # process_weight: [H, H] (no bias); transpose to [K=H, N=H] = W_T
        W_T = process_weight.t().contiguous()

        # Output tensors (float32 for computation)
        out_cat = torch.empty((B, L, H), device=hidden_states.device, dtype=torch.float32)
        processed = torch.empty((B, L, H), device=hidden_states.device, dtype=torch.float32)

        # Strides
        stride_ob = out_cat.stride(0); stride_ol = out_cat.stride(1); stride_oh = out_cat.stride(2)
        stride_eb = enc.stride(0); stride_et = enc.stride(1); stride_eh = enc.stride(2)
        stride_hb = hid.stride(0); stride_hi = hid.stride(1); stride_hh = hid.stride(2)

        # Launch Triton concat kernel: grid over (batch, tiles over L, tiles over H)
        BLOCK_M = 128  # tile over L
        BLOCK_K = 128  # tile over H
        grid_cat = (B, triton.cdiv(L, BLOCK_M), triton.cdiv(H, BLOCK_K))
        concat_encoder_hidden_kernel[grid_cat](
            out_cat, enc, hid,
            B, T, I, H,
            stride_ob, stride_ol, stride_oh,
            stride_eb, stride_et, stride_eh,
            stride_hb, stride_hi, stride_hh,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        )

        # Strides for GEMM
        stride_ab = out_cat.stride(0)  # A: [B, M=L, K=H]
        stride_am = out_cat.stride(1)
        stride_ak = out_cat.stride(2)
        stride_bb = W_T.stride(0)      # B: [K=H, N=H]
        stride_bk = W_T.stride(1)
        stride_bn = W_T.stride(2)
        stride_cb = processed.stride(0)
        stride_cm = processed.stride(1)
        stride_cn = processed.stride(2)

        # Launch Triton GEMM kernel with fixed tile sizes (no autotune to avoid grid mismatches)
        BLOCK_M_GEMM = 64
        BLOCK_N_GEMM = 64
        BLOCK_K_GEMM = 32
        grid_gemm = (B, triton.cdiv(L, BLOCK_M_GEMM), triton.cdiv(H, BLOCK_N_GEMM))
        batched_gemm_kernel[grid_gemm](
            out_cat, W_T, processed,
            BATCH=B, M=L, N=H, K=H,
            BLOCK_M=BLOCK_M_GEMM, BLOCK_N=BLOCK_N_GEMM, BLOCK_K=BLOCK_K_GEMM,
        )

        # Split outputs
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
