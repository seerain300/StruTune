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
    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        # Load Kc and Kp chunks
        Kc_chunk = tl.load(Kc_ptr + t_idx[:, None] * Dq + tl.arange(0, Dq), mask=mask_t[:, None], other=0.0)  # [BLOCK_T, Dq]
        Kp_chunk = tl.load(Kp_ptr + t_idx[:, None] * Dp + tl.arange(0, Dp), mask=mask_t[:, None], other=0.0)  # [BLOCK_T, Dp]
        # Dot products: sum over dim=1
        dot1 = tl.sum(qn[None, :] * Kc_chunk, axis=1)  # [BLOCK_T]
        dot2 = tl.sum(qp[None, :] * Kp_chunk, axis=1)  # [BLOCK_T]
        acc += dot1 + dot2

    # Store logits for this head into logits_ptr[h, t_start:t_start+BLOCK_T]
    store_mask = tl.arange(0, BLOCK_T) < T
    tl.store(logits_ptr + h * T + t_start + tl.arange(0, BLOCK_T), acc, mask=store_mask)


# Triton kernel: row-wise softmax over T for logits[h, :]
# Inputs:
#   logits_ptr: [H, T], float32
#   soft_ptr:   [H, T], float32 (output)
@triton.jit
def softmax_row_kernel(logits_ptr, soft_ptr, H: tl.constexpr, T: tl.constexpr, BLOCK_T: tl.constexpr):
    h = tl.program_id(0)
    offs_t = tl.arange(0, BLOCK_T)
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        row = tl.load(logits_ptr + h * T + t_idx, mask=mask_t, other=-float('inf'))  # [BLOCK_T]
        # Compute max for numerical stability
        row_max = tl.max(row, axis=0)  # scalar
        exp_row = tl.exp(row - row_max)  # [BLOCK_T]
        denom = tl.sum(exp_row, axis=0)  # scalar
        soft_row = exp_row / denom  # [BLOCK_T]
        tl.store(soft_ptr + h * T + t_idx, soft_row, mask=mask_t)


# Triton kernel: row-wise logsumexp for logits[h, :], write logsumexp(h) / ln(2) to lse_ptr[h]
# Inputs:
#   logits_ptr: [H, T], float32
#   lse_ptr:    [H], float32 (output)
@triton.jit
def lse_row_kernel(logits_ptr, lse_ptr, H: tl.constexpr, T: tl.constexpr, BLOCK_T: tl.constexpr):
    h = tl.program_id(0)
    offs_t = tl.arange(0, BLOCK_T)

    # Pass 1: compute row_max
    row_max = -float('inf')
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        row = tl.load(logits_ptr + h * T + t_idx, mask=mask_t, other=-float('inf'))  # [BLOCK_T]
        tile_max = tl.max(row, axis=0)
        row_max = tl.maximum(row_max, tile_max)

    # Pass 2: compute sum_exp
    sum_exp = 0.0
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        row = tl.load(logits_ptr + h * T + t_idx, mask=mask_t, other=-float('inf'))  # [BLOCK_T]
        sum_exp += tl.sum(tl.exp(row - row_max), axis=0)

    lse_val = tl.log(sum_exp) + row_max  # logsumexp
    ln2 = 0.6931471805599453
    tl.store(lse_ptr + h, lse_val / ln2)


# Triton kernel: out[h, d] = sum_t soft[h, t] * Kc[t, d]
# Inputs:
#   soft_ptr:   [H, T], float32 (attention weights, softmax of scaled logits)
#   Kc_ptr:     [T, Dq], float32
#   out_ptr:    [H, Dq], float32 (output)
@triton.jit
def attn_matmul_kernel(soft_ptr, Kc_ptr, out_ptr,
                        H: tl.constexpr, T: tl.constexpr, Dq: tl.constexpr,
                        BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
    h = tl.program_id(0)  # one program per head
    offs_t = tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_D)

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        attn_chunk = tl.load(soft_ptr + h * T + t_idx, mask=mask_t, other=0.0)  # [BLOCK_T]
        Kc_chunk = tl.load(Kc_ptr + t_idx[:, None] * Dq + offs_d[None, :],
                           mask=mask_t[:, None],
                           other=0.0)  # [BLOCK_T, BLOCK_D]
        # Unroll over BLOCK_T to accumulate
        for t in range(0, BLOCK_T):
            tt = t_start + t
            if tt < T:
                acc += attn_chunk[t] * Kc_chunk[t, :]

    tl.store(out_ptr + h * Dq + offs_d, acc, mask=offs_d < Dq)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # constants from the original assertions
        self.num_qo_heads = 16
        self.head_dim_ckv = 512  # Dq
        self.head_dim_kpe = 64    # Dp

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA and dtype float32 for compute
        device = q_nope.device
        use_cuda = device.type == 'cuda'
        if not use_cuda:
            device = torch.device('cuda')
        q_nope = q_nope.to(device)
        q_pe = q_pe.to(device)
        ckv_cache = ckv_cache.to(device)
        kpe_cache = kpe_cache.to(device)
        kv_indptr = kv_indptr.to(device)
        kv_indices = kv_indices.to(device)

        # Cast to float32 for Triton compute
        q_nope_f32 = q_nope.contiguous().to(torch.float32)   # [B, 16, 512]
        q_pe_f32 = q_pe.contiguous().to(torch.float32)       # [B, 16, 64]
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [N, 512]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [N, 64]

        B = q_nope_f32.shape[0]
        H = self.num_qo_heads
        Dq = self.head_dim_ckv
        Dp = self.head_dim_kpe

        output = torch.empty((B, H, Dq), dtype=torch.float32, device=device)  # [B, 16, 512], float32 for compute
        lse = torch.empty((B, H), dtype=torch.float32, device=device)          # [B, 16]

        for b in range(B):
            # Compute L_tokens from kv_indptr
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L_tokens = max(page_end - page_beg, 0)
            if L_tokens == 0:
                # No tokens used for this batch element; set output and lse to zeros
                output[b].zero_()
                lse[b].zero_()
                continue

            # Gather used tokens
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32)  # [L_tokens]
            # Gather Kc and Kp vectors for used tokens
            Kc = Kc_all[tok_idx]  # [L_tokens, 512]
            Kp = Kp_all[tok_idx]  # [L_tokens, 64]

            # Preload q vectors for this batch element (cast to float32 for compute)
            qn = q_nope_f32[b]   # [16, 512]
            qp = q_pe_f32[b]     # [16, 64]

            # Allocate logits [H, T]
            logits = torch.empty((H, L_tokens), dtype=torch.float32, device=device)

            # Launch fused_logits_kernel: one program per head
            BLOCK_T = 256
            grid_logits = (H,)
            fused_logits_kernel[grid_logits](
                qn, qp, Kc, Kp, logits,
                H=H, T=L_tokens, Dq=Dq, Dp=Dp,
                BLOCK_T=BLOCK_T,
            )

            # Scale logits
            scaled = logits * sm_scale

            # Compute softmax per row (head) into soft [H, T]
            soft = torch.empty((H, L_tokens), dtype=torch.float32, device=device)
            BLOCK_T_SOFT = 256
            grid_softmax = (H,)
            softmax_row_kernel[grid_softmax](
                scaled, soft,
                H=H, T=L_tokens, BLOCK_T=BLOCK_T_SOFT,
            )

            # Compute LSE per head and divide by ln(2)
            grid_lse = (H,)
            lse_row = torch.empty((H,), dtype=torch.float32, device=device)
            lse_row_kernel[grid_lse](
                scaled, lse_row,
                H=H, T=L_tokens, BLOCK_T=BLOCK_T_SOFT,
            )
            lse[b] = lse_row  # [16]

            # Compute output per head: out[h] = soft[h] @ Kc
            for h in range(H):
                out_h = torch.empty((Dq,), dtype=torch.float32, device=device)
                attn_matmul_kernel[(1,)](
                    soft[h], Kc, out_h,
                    H=1, T=L_tokens, Dq=Dq,
                    BLOCK_T=BLOCK_T_SOFT, BLOCK_D=128,
                )
                output[b, h] = out_h

        # Return output as bfloat16 to match original, and lse as float32
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
