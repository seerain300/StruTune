import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute logits[h, t] = dot(qn[h, :], Kc[t, :]) + dot(qp[h, :], Kp[t, :])
# Inputs:
#   q_nope_ptr: [H, Dq], float32
#   q_pe_ptr:   [H, Dp], float32
#   Kc_ptr:     [T, Dq], float32
#   Kp_ptr:     [T, Dp], float32
#   logits_ptr: [H, T], float32 (output)
@triton.jit
def fused_logits_kernel(q_nope_ptr, q_pe_ptr, Kc_ptr, Kp_ptr, logits_ptr,
                         H: tl.constexpr, T: tl.constexpr, Dq: tl.constexpr, Dp: tl.constexpr,
                         BLOCK_T: tl.constexpr):
    h = tl.program_id(0)  # one program per head
    # Preload q vectors for this head using compile-time Dq/Dp
    qn = tl.load(q_nope_ptr + h * Dq + tl.arange(0, Dq))  # [Dq]
    qp = tl.load(q_pe_ptr + h * Dp + tl.arange(0, Dp))    # [Dp]

    offs_t = tl.arange(0, BLOCK_T)
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        # Accumulate dot products for this tile
        acc = tl.zeros([BLOCK_T], dtype=tl.float32)
        # Dot over Kc (Dq dim)
        for i in range(0, Dq):
            kqi = tl.load(Kc_ptr + t_idx * Dq + i, mask=mask_t, other=0.0)  # [BLOCK_T]
            acc += qn[i] * kqi
        # Dot over Kp (Dp dim)
        for i in range(0, Dp):
            kpj = tl.load(Kp_ptr + t_idx * Dp + i, mask=mask_t, other=0.0)  # [BLOCK_T]
            acc += qp[i] * kpj
        # Store logits for this head and tile
        tl.store(logits_ptr + h * T + t_idx, acc, mask=mask_t)


# Triton kernel: row-wise softmax over the last dimension for scaled_logits[h, :]
# Inputs:
#   scaled_ptr: [H, T], float32 (input scaled logits)
#   attn_ptr:   [H, T], float32 (output attn)
@triton.jit
def softmax_row_kernel(scaled_ptr, attn_ptr,
                        H: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
                        BLOCK_T: tl.constexpr):
    h = tl.program_id(0)
    offs_t = tl.arange(0, BLOCK_T)
    # First pass: compute max for numerical stability
    m = -float("inf")
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        x = tl.load(scaled_ptr + h * T + t_idx, mask=mask_t, other=-float("inf"))
        # reduce max across this tile
        tile_max = tl.max(x, axis=0)
        m = tl.maximum(m, tile_max)
    # Second pass: compute denominator sum(exp(x - m))
    denom = 0.0
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        x = tl.load(scaled_ptr + h * T + t_idx, mask=mask_t, other=-float("inf"))
        e = tl.exp(x - m)
        denom += tl.sum(e, axis=0)
    inv_denom = 1.0 / denom
    # Third pass: write normalized attn
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        x = tl.load(scaled_ptr + h * T + t_idx, mask=mask_t, other=-float("inf"))
        e = tl.exp(x - m) * inv_denom
        tl.store(attn_ptr + h * T + t_idx, e, mask=mask_t)


# Triton kernel: row-wise logsumexp for scaled_logits[h, :], store to out[h] = logsumexp / ln(2)
# Inputs:
#   scaled_ptr: [H, T], float32
#   out_ptr:    [H], float32
@triton.jit
def lse_row_kernel(scaled_ptr, out_ptr,
                   H: tl.constexpr, T: tl.constexpr,
                   BLOCK_T: tl.constexpr):
    h = tl.program_id(0)
    offs_t = tl.arange(0, BLOCK_T)
    m = -float("inf")
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        x = tl.load(scaled_ptr + h * T + t_idx, mask=mask_t, other=-float("inf"))
        tile_max = tl.max(x, axis=0)
        m = tl.maximum(m, tile_max)
    sum_exp = 0.0
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        x = tl.load(scaled_ptr + h * T + t_idx, mask=mask_t, other=-float("inf"))
        sum_exp += tl.sum(tl.exp(x - m), axis=0)
    lse = tl.log(sum_exp) + m  # logsumexp
    ln2 = 0.6931471805599453  # math.log(2)
    tl.store(out_ptr + h, lse / ln2)


# Triton kernel: compute out[h, d] = sum_t attn[h, t] * Kc[t, d]
# Inputs:
#   attn_ptr:   [H, T], float32
#   Kc_ptr:     [T, D], float32
#   out_ptr:    [H, D], float32 (output)
@triton.jit
def attn_matmul_kernel(attn_ptr, Kc_ptr, out_ptr,
                        H: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
                        BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
    h = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    for d_start in range(0, D, BLOCK_D):
        d_idx = d_start + offs_d
        mask_d = d_idx < D
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for t_start in range(0, T, BLOCK_T):
            t_idx = t_start + tl.arange(0, BLOCK_T)
            mask_t = t_idx < T
            # Load attn chunk [BLOCK_T]
            attn_sub = tl.load(attn_ptr + h * T + t_idx, mask=mask_t, other=0.0)  # [BLOCK_T]
            # Load Kc chunk [BLOCK_T, BLOCK_D]
            Kc_chunk = tl.load(
                Kc_ptr + t_idx[:, None] * D + d_idx[None, :],
                mask=mask_t[:, None] & mask_d[None, :],
                other=0.0
            )  # [BLOCK_T, BLOCK_D]
            # Accumulate over tokens in this tile
            acc += tl.sum(attn_sub[:, None] * Kc_chunk, axis=0)
        tl.store(out_ptr + h * D + d_idx, acc, mask=mask_d)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA and float32 compute
        device = q_nope.device
        if device.type != 'cuda':
            device = torch.device('cuda')
            q_nope = q_nope.to(device)
            q_pe = q_pe.to(device)
            ckv_cache = ckv_cache.to(device)
            kpe_cache = kpe_cache.to(device)
            kv_indptr = kv_indptr.to(device)
            kv_indices = kv_indices.to(device)
        q_nope_f32 = q_nope.contiguous().to(torch.float32)
        q_pe_f32 = q_pe.contiguous().to(torch.float32)
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [N, 512]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [N, 64]

        B = q_nope_f32.shape[0]
        H = q_nope_f32.shape[1]
        Dq = q_nope_f32.shape[2]
        Dp = q_pe_f32.shape[2]

        # Prepare outputs
        output = torch.empty((B, H, Dq), dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Constants for tiling
        BLOCK_T = 256
        BLOCK_D = 128

        for b in range(B):
            # Determine used tokens range
            if int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item()) <= 0:
                # No valid tokens for this batch element: output zeros
                output[b].zero_()
                lse[b].zero_()
                continue

            # Gather Kc and Kp for this batch element
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            tok_idx = kv_indices[start:end].contiguous()  # [T]
            T = tok_idx.numel()
            Kc = Kc_all[tok_idx]  # [T, Dq]
            Kp = Kp_all[tok_idx]  # [T, Dp]

            # Q vectors for this batch
            qn = q_nope_f32[b]  # [H, Dq]
            qp = q_pe_f32[b]    # [H, Dp]

            # 1) Compute logits[h, t] = qn[h] @ Kc[t] + qp[h] @ Kp[t]
            logits = torch.empty((H, T), dtype=torch.float32, device=device)
            # Launch fused_logits_kernel: one program per head
            grid_logits = (H,)
            fused_logits_kernel[grid_logits](
                qn, qp, Kc, Kp, logits,
                H=H, T=T, Dq=Dq, Dp=Dp,
                BLOCK_T=BLOCK_T
            )

            # 2) Compute lse per head: logsumexp(logits_scaled) / ln(2)
            scaled = logits * sm_scale
            # Launch lse_row_kernel: one program per head
            grid_lse = (H,)
            lse_row_kernel[grid_lse](
                scaled, lse[b],
                H=H, T=T,
                BLOCK_T=BLOCK_T
            )

            # 3) Compute attention: softmax(scaled, dim=-1)
            attn = torch.empty((H, T), dtype=torch.float32, device=device)
            grid_softmax = (H,)
            softmax_row_kernel[grid_softmax](
                scaled, attn,
                H=H, T=T, D=T,  # D==T, we can reuse T for D
                BLOCK_T=BLOCK_T
            )

            # 4) Compute output[h, :] = attn[h, :] @ Kc[:, :]
            for h_idx in range(H):
                out_vec = torch.empty((Dq,), dtype=torch.float32, device=device)
                attn_matmul_kernel[(1,)](
                    attn[h_idx], Kc, out_vec,
                    H=1, T=T, D=Dq,
                    BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D
                )
                output[b, h_idx] = out_vec

        # Cast output back to bfloat16 to match original; lse remains float32
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
