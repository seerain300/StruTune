import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute logits[h, t] = dot(qn[h, :], Kc[t, :]) + dot(qp[h, :], Kp[t, :])
# q_nope: [H, Dq], q_pe: [H, Dp], Kc: [T, Dq], Kp: [T, Dp], logits: [H, T]
@triton.jit
def fused_logits_kernel(q_nope_ptr, q_pe_ptr, Kc_ptr, Kp_ptr, logits_ptr,
                         H: tl.constexpr, T: tl.constexpr, Dq: tl.constexpr, Dp: tl.constexpr,
                         BLOCK_T: tl.constexpr):
    h = tl.program_id(0)  # one program per head
    # Preload q vectors for this head using compile-time Dq/Dp
    qn = tl.load(q_nope_ptr + h * Dq + tl.arange(0, Dq))  # [Dq]
    qp = tl.load(q_pe_ptr + h * Dp + tl.arange(0, Dp))    # [Dp]

    offs_t = tl.arange(0, BLOCK_T)

    # Loop over tokens in tiles of size BLOCK_T
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t  # [BLOCK_T]
        mask_t = t_idx < T

        # Load Kc_sub [BLOCK_T, Dq] and Kp_sub [BLOCK_T, Dp]
        Kc_sub = tl.load(Kc_ptr + t_idx[:, None] * Dq + tl.arange(0, Dq), mask=mask_t[:, None], other=0.0)  # [BLOCK_T, Dq]
        Kp_sub = tl.load(Kp_ptr + t_idx[:, None] * Dp + tl.arange(0, Dp), mask=mask_t[:, None], other=0.0)  # [BLOCK_T, Dp]

        # Compute dot-products: sum over feature dims
        # sum_j Kc_sub[t, j] * qn[j]
        dot1 = tl.sum(Kc_sub * qn[None, :], axis=1)  # [BLOCK_T]
        # sum_j Kp_sub[t, j] * qp[j]
        dot2 = tl.sum(Kp_sub * qp[None, :], axis=1)  # [BLOCK_T]

        logits_row = dot1 + dot2  # [BLOCK_T]
        # Store into logits[h, t_start : t_start + BLOCK_T]
        tl.store(logits_ptr + h * T + t_start + offs_t, logits_row, mask=mask_t)


# Triton kernel: row-wise softmax over logits[h, :]
# logits_ptr: [H, T], out_ptr: [H, T], T: tl.constexpr
@triton.jit
def softmax_row_kernel(logits_ptr, attn_ptr, T: tl.constexpr, BLOCK_T: tl.constexpr):
    h = tl.program_id(0)  # one program per head
    offs = tl.arange(0, BLOCK_T)
    # Load logits row
    row = tl.load(logits_ptr + h * T + offs)
    # Stable softmax
    m = tl.max(row, axis=0)
    row_shift = row - m
    exp_row = tl.exp(row_shift)
    den = tl.sum(exp_row, axis=0)
    attn = exp_row / den
    tl.store(attn_ptr + h * T + offs, attn)


# Triton kernel: row-wise logsumexp over logits[h, :]
# logits_ptr: [H, T], lse_ptr: [H], T: tl.constexpr
@triton.jit
def lse_row_kernel(logits_ptr, lse_ptr, T: tl.constexpr, BLOCK_T: tl.constexpr):
    h = tl.program_id(0)  # one program per head
    offs = tl.arange(0, BLOCK_T)
    # Iterate over tokens in tiles and compute max
    m = -float('inf')
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs
        mask_t = t_idx < T
        row = tl.load(logits_ptr + h * T + t_idx, mask=mask_t, other=-float('inf'))
        m = tl.maximum(m, tl.max(row, axis=0))
    # Second pass: sum exp(logits - m)
    s = 0.0
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs
        mask_t = t_idx < T
        row = tl.load(logits_ptr + h * T + t_idx, mask=mask_t, other=-float('inf'))
        s += tl.sum(tl.exp(row - m), axis=0)
    # LSE = m + log(s), divided by ln(2)
    lse_val = m + tl.log(s)  # Triton provides tl.log
    # ln(2) constant
    ln2 = 0.6931471805599453
    tl.store(lse_ptr + h, lse_val / ln2)


# Triton kernel: per-head matmul out[h, :] = attn[h, :] @ Kc[:, :]
# attn_ptr: [H, T], Kc_ptr: [T, Dq], out_ptr: [H, Dq]
@triton.jit
def matmul_row_kernel(attn_ptr, Kc_ptr, out_ptr,
                      H: tl.constexpr, T: tl.constexpr, Dq: tl.constexpr,
                      BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
    h = tl.program_id(0)  # one program per head
    # Load qn for this head (same as attn * Kc could be done in PyTorch, but here we implement matmul)
    # We won't need qn here; we'll compute out[h, :] = sum_t attn[h, t] * Kc[t, :]
    # We'll do it in tiles over T to keep register usage controlled.
    # Initialize output vector
    # Allocate out vector (we assume out_ptr is already allocated in host code as [H, Dq])
    # We'll perform accumulation in float32.
    # Note: Triton does not support storing partials across tiles easily without an intermediate.
    # We'll loop over T in tiles, compute partial sums for each D-dimension in BLOCK_D chunks, and accumulate.
    # However, a simpler and efficient approach is to use PyTorch for matmul here (original allowed),
    # but to stay strictly Triton-only, we implement a tiled accumulation using loops.
    # For simplicity and performance, we implement a tiled reduction over T for each d in Dq.
    # This kernel is more involved; to keep code size reasonable and correctness, we implement a basic tiled reduction:
    # out_vec = [0]*Dq
    # For d in range(0, Dq): out_vec[d] = sum_t attn[h, t] * Kc[t, d]
    # Since Triton lacks direct vector init to float, we compute each d via loop:
    for d in range(0, Dq):
        acc = 0.0
        for t_start in range(0, T, BLOCK_T):
            t_idx = t_start + tl.arange(0, BLOCK_T)
            mask_t = t_idx < T
            attn_row = tl.load(attn_ptr + h * T + t_idx, mask=mask_t, other=0.0)  # [BLOCK_T]
            Kc_col_d = tl.load(Kc_ptr + t_idx * Dq + d, mask=mask_t, other=0.0)  # [BLOCK_T]
            acc += tl.sum(attn_row * Kc_col_d, axis=0)
        tl.store(out_ptr + h * Dq + d, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed constants from the original constraints
        self.Dq = 512  # head_dim_ckv
        self.Dp = 64   # head_dim_kpe
        self.H = 16    # num_qo_heads
        # Power-of-two block sizes (to satisfy Triton's arange constraint)
        self.BLOCK_T = 128
        self.BLOCK_D = 128  # for reductions, though we loop over Dq

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale, use_triton=True):
        # Ignore extra 'use_triton' argument; keep signature compatible with evaluation harness.
        device = q_nope.device
        batch_size = q_nope.shape[0]

        # Prepare outputs
        output = torch.empty((batch_size, self.H, self.Dq), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, self.H), dtype=torch.float32, device=device)

        for b in range(batch_size):
            # Compute L_tokens and tok_idx
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No KV tokens for this batch element; match original behavior
                # Original sets output[b] to zeros and lse[b] to -inf (we set zeros for output, lse to -inf)
                output[b].zero_()
                lse[b].fill_(-float("inf"))
                continue

            tok_idx = kv_indices[kv_indptr[b].item(): kv_indptr[b + 1].item()].to(torch.int64)
            # Load Kc and Kp for this batch element, squeeze to remove [1]
            Kc_b = ckv_cache[tok_idx]   # [L_tokens, self.Dq], bfloat16
            Kp_b = kpe_cache[tok_idx]   # [L_tokens, self.Dp], bfloat16

            # Load q vectors for this batch element
            qn = q_nope[b]  # [self.H, self.Dq], bfloat16
            qp = q_pe[b]    # [self.H, self.Dp], bfloat16

            # Cast to float32 for Triton compute
            Kc_b = Kc_b.to(torch.float32)  # [L_tokens, 512]
            Kp_b = Kp_b.to(torch.float32)  # [L_tokens, 64]
            qn = qn.to(torch.float32)      # [16, 512]
            qp = qp.to(torch.float32)      # [16, 64]

            # Ensure tensors are contiguous
            Kc_b = Kc_b.contiguous()
            Kp_b = Kp_b.contiguous()
            qn = qn.contiguous()
            qp = qp.contiguous()

            # Allocate logits [H, T] in float32
            logits = torch.empty((self.H, L_tokens), dtype=torch.float32, device=device)

            # Launch fused_logits_kernel: one program per head
            grid = (self.H,)
            fused_logits_kernel[grid](
                qn, qp, Kc_b, Kp_b, logits,
                H=self.H, T=L_tokens, Dq=self.Dq, Dp=self.Dp,
                BLOCK_T=self.BLOCK_T,
                num_warps=4, num_stages=2
            )

            # Compute scaled logits
            logits_scaled = logits * sm_scale

            # Compute lse[h] per head using Triton kernel (row-wise logsumexp)
            grid_lse = (self.H,)
            lse[b] = torch.full((self.H,), -float("inf"), dtype=torch.float32, device=device)
            lse_row_kernel[grid_lse](
                logits_scaled, lse[b],
                T=L_tokens, BLOCK_T=self.BLOCK_T,
                num_warps=4, num_stages=2
            )

            # Compute attn[h, :] = softmax(logits_scaled[h, :]) using Triton kernel
            attn = torch.empty((self.H, L_tokens), dtype=torch.float32, device=device)
            softmax_row_kernel[grid](
                logits_scaled, attn,
                T=L_tokens, BLOCK_T=self.BLOCK_T,
                num_warps=4, num_stages=2
            )

            # Compute output[h, :] = attn[h, :] @ Kc_b[:, :] using Triton matmul_row_kernel
            # out_ptr is [H, Dq], we'll allocate and fill
            out_vec = torch.empty((self.H, self.Dq), dtype=torch.float32, device=device)
            matmul_row_kernel[grid](
                attn, Kc_b, out_vec,
                H=self.H, T=L_tokens, Dq=self.Dq,
                BLOCK_T=self.BLOCK_T, BLOCK_D=self.BLOCK_D,
                num_warps=4, num_stages=2
            )

            # Assign to output[b, :, :]
            output[b] = out_vec

        # Convert output to bfloat16 to match original output dtype
        output = output.to(torch.bfloat16)

        # Return output and lse (lse is float32 as in original)
        return output, lse


def run(*args):
    return ModelNew()(*args)
