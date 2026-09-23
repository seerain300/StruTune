import torch
import triton
import triton.language as tl


# Fused logits kernel: compute logits[h, t] = qn[h, :] @ Kc[t, :] + qp[h, :] @ Kp[t, :]
@triton.jit
def fused_logits_kernel(
    qn_ptr, qp_ptr, Kc_ptr, Kp_ptr, logits_ptr,
    H: tl.constexpr,  # 16
    Dq: tl.constexpr, # 512
    Dp: tl.constexpr, # 64
    T: tl.constexpr,  # number of tokens (compile-time for unrolling)
    BLOCK_T: tl.constexpr,  # tile size along tokens (power-of-two)
):
    h = tl.program_id(0)  # one program per head
    # Preload qn[h, :] and qp[h, :]
    # qn[h, :] -> contiguous vector of length Dq
    # qp[h, :] -> contiguous vector of length Dp
    # We'll compute logits for tiles of tokens
    for t_start in range(0, T, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        # accumulators for the two dot-products
        acc1 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_T,), dtype=tl.float32)
        # Loop over features (Dq and Dp are constexpr, Triton can unroll)
        for d in range(0, Dq):
            qn_d = tl.load(qn_ptr + h * Dq + d)  # qn[h, d]
            kc_ptr_d = Kc_ptr + offs_t * Dq + d
            kc_vals = tl.load(kc_ptr_d, mask=mask_t, other=0.0)
            acc1 += qn_d * kc_vals
        for d in range(0, Dp):
            qp_d = tl.load(qp_ptr + h * Dp + d)  # qp[h, d]
            kp_ptr_d = Kp_ptr + offs_t * Dp + d
            kp_vals = tl.load(kp_ptr_d, mask=mask_t, other=0.0)
            acc2 += qp_d * kp_vals
        logits_tile = acc1 + acc2
        # store logits[h, offs_t]
        tl.store(logits_ptr + h * T + offs_t, logits_tile, mask=mask_t)


# Softmax per row (head) in Triton: attn[h, :] = softmax(logits_scaled[h, :])
@triton.jit
def softmax_row_kernel(logits_ptr, attn_ptr, T: tl.constexpr, BLOCK_T: tl.constexpr, sm_scale: tl.float32):
    h = tl.program_id(0)  # one program per head
    # First pass: compute max for numerical stability
    max_val = tl.full((), -float('inf'), tl.float32)
    for t_start in range(0, T, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        vals = tl.load(logits_ptr + h * T + offs_t, mask=mask_t, other=-float('inf'))
        # scale
        vals = vals * sm_scale
        # reduce to find max
        tile_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, tile_max)
    # Second pass: compute sum of exp(vals - max)
    sum_exp = tl.zeros((), dtype=tl.float32)
    for t_start in range(0, T, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        vals = tl.load(logits_ptr + h * T + offs_t, mask=mask_t, other=-float('inf'))
        vals = vals * sm_scale
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(exp_vals, axis=0)
    # Third pass: write normalized attn
    for t_start in range(0, T, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        vals = tl.load(logits_ptr + h * T + offs_t, mask=mask_t, other=-float('inf'))
        vals = vals * sm_scale
        attn_vals = tl.exp(vals - max_val) / sum_exp
        tl.store(attn_ptr + h * T + offs_t, attn_vals, mask=mask_t)


# Logsumexp per row (head) in Triton: lse[h] = logsumexp(logits_scaled[h, :]) / ln(2)
@triton.jit
def lse_row_kernel(logits_ptr, lse_ptr, T: tl.constexpr, BLOCK_T: tl.constexpr, sm_scale: tl.float32):
    h = tl.program_id(0)  # one program per head
    max_val = tl.full((), -float('inf'), tl.float32)
    for t_start in range(0, T, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        vals = tl.load(logits_ptr + h * T + offs_t, mask=mask_t, other=-float('inf'))
        vals = vals * sm_scale
        tile_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, tile_max)
    sum_exp = tl.zeros((), dtype=tl.float32)
    for t_start in range(0, T, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        vals = tl.load(logits_ptr + h * T + offs_t, mask=mask_t, other=-float('inf'))
        vals = vals * sm_scale
        sum_exp += tl.sum(tl.exp(vals - max_val), axis=0)
    lse_val = max_val + tl.log(sum_exp)  # logsumexp
    # ln(2) = 0.6931471805599453
    lse_val = lse_val / 0.6931471805599453
    tl.store(lse_ptr + h, lse_val)


# Per-head matmul in Triton: out[h, :] = attn[h, :] @ Kc[:, :]
# We implement a simple kernel with a compile-time-unrolled loop over tokens.
@triton.jit
def matmul_row_kernel(attn_ptr, Kc_ptr, out_ptr,
                      Dq: tl.constexpr, T: tl.constexpr,
                      BLOCK_D: tl.constexpr):
    h = tl.program_id(0)  # grid = (H,) — one program per head
    # out_ptr points to the base of output[h, :], which is a contiguous vector of length Dq
    # Initialize out_vec
    out_vec = tl.zeros((Dq,), dtype=tl.float32)
    # Iterate over tokens in compile-time-unrolled loop
    for t in range(0, T):
        attn_t = tl.load(attn_ptr + h * T + t)  # scalar
        kc_row = tl.load(Kc_ptr + t * Dq + tl.arange(0, BLOCK_D))  # vector of length Dq (BLOCK_D can be <= Dq; here we set BLOCK_D == Dq)
        out_vec += attn_t * kc_row
    # store out[h, :] (contiguous vector of length Dq)
    offs_d = tl.arange(0, Dq)
    tl.store(out_ptr + offs_d, out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # constants for this workload
        self.H = 16
        self.Dq = 512
        self.Dp = 64
        # tile sizes (power-of-two)
        self.BLOCK_T = 128  # tokens tile
        self.BLOCK_D = 64    # feature tile for matmul (here equals Dq, but we still keep it as constexpr)

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale, extra_flag):
        # ignore extra_flag (to match harness signature), only use tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA"
        B = q_nope.shape[0]
        device = q_nope.device

        # Output and lse tensors
        output = torch.empty((B, self.H, self.Dq), dtype=torch.float32, device=device)  # [B, 16, 512] float32
        lse = torch.empty((B, self.H), dtype=torch.float32, device=device)

        for b in range(B):
            # Compute L_tokens and tok_idx
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No valid tokens, zero output and lse
                output[b].zero_()
                lse[b].zero_()
                continue

            # Gather Kc and Kp for this batch element from cache
            # tok_idx is a slice of kv_indices between [kv_indptr[b], kv_indptr[b+1))
            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]].to(torch.int32)
            Kc_b = ckv_cache[tok_idx].squeeze(1).to(torch.float32)  # [L_tokens, 512]
            Kp_b = kpe_cache[tok_idx].squeeze(1).to(torch.float32)  # [L_tokens, 64]

            # qn and qp
            qn = q_nope[b].to(torch.float32)  # [16, 512]
            qp = q_pe[b].to(torch.float32)    # [16, 64]

            # Allocate logits [H, T]
            logits = torch.empty((self.H, L_tokens), dtype=torch.float32, device=device)

            # Launch fused_logits_kernel: one program per head
            grid_logits = (self.H,)
            fused_logits_kernel[grid_logits](
                qn, qp, Kc_b, Kp_b, logits,
                H=self.H, Dq=self.Dq, Dp=self.Dp, T=L_tokens, BLOCK_T=self.BLOCK_T,
                num_warps=4, num_stages=2
            )

            # Compute attn per head via Triton softmax
            attn = torch.empty((self.H, L_tokens), dtype=torch.float32, device=device)
            grid_softmax = (self.H,)
            softmax_row_kernel[grid_softmax](
                logits, attn,
                T=L_tokens, BLOCK_T=self.BLOCK_T, sm_scale=float(sm_scale),
                num_warps=4, num_stages=2
            )

            # Compute lse per head via Triton
            grid_lse = (self.H,)
            lse_row_kernel[grid_lse](
                logits, lse[b],
                T=L_tokens, BLOCK_T=self.BLOCK_T, sm_scale=float(sm_scale),
                num_warps=4, num_stages=2
            )

            # Compute per-head matmul out[h, :] = attn[h, :] @ Kc_b[:, :] using Triton
            # Launch one program per head to write its output row
            grid_matmul = (self.H,)
            matmul_row_kernel[grid_matmul](
                attn, Kc_b, output[b],  # output[b] is [16, 512], contiguous
                Dq=self.Dq, T=L_tokens, BLOCK_D=self.BLOCK_D,
                num_warps=4, num_stages=2
            )

        return output, lse


# The original helper functions can remain as-is. The evaluation harness will call ModelNew.forward with 8 positional arguments.
# We ignore the extra flag in forward.


def run(*args):
    return ModelNew()(*args)
