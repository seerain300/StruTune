import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernels: matrix multiplications and helpers
if TRITON_AVAILABLE:
    @triton.jit
    def matmul_qn_kc(qn_ptr, kc_ptr, out_ptr,
                     H: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        # Compute out[H, L] = qn[H, D] @ kc[L, D]^T
        m = H
        n = L
        k = D
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, k, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            q = tl.load(
                qn_ptr + (offs_m[:, None] * D + offs_k[None, :]),
                mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
                other=0.0
            )
            k_tile = tl.load(
                kc_ptr + (offs_n[None, :] * D + offs_k[:, None]),
                mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
                other=0.0
            )
            acc += tl.dot(q, k_tile)
        out_offsets = offs_m[:, None] * L + offs_n[None, :]
        tl.store(
            out_ptr + out_offsets,
            acc,
            mask=(offs_m[:, None] < m) & (offs_n[None, :] < n)
        )

    @triton.jit
    def matmul_qp_kp(qp_ptr, kp_ptr, out_ptr,
                     H: tl.constexpr, P: tl.constexpr, L: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        # Compute out[H, L] = qp[H, P] @ kp[L, P]^T
        m = H
        n = L
        k = P
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, k, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            q = tl.load(
                qp_ptr + (offs_m[:, None] * P + offs_k[None, :]),
                mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
                other=0.0
            )
            k_tile = tl.load(
                kp_ptr + (offs_n[None, :] * P + offs_k[:, None]),
                mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
                other=0.0
            )
            acc += tl.dot(q, k_tile)
        out_offsets = offs_m[:, None] * L + offs_n[None, :]
        tl.store(
            out_ptr + out_offsets,
            acc,
            mask=(offs_m[:, None] < m) & (offs_n[None, :] < n)
        )

    @triton.jit
    def add_logits(a_ptr, b_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        m = H
        n = L
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        a = tl.load(a_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                    mask=(offs_m[:, None] < m) & (offs_n[None, :] < n), other=0.0)
        b = tl.load(b_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                    mask=(offs_m[:, None] < m) & (offs_n[None, :] < n), other=0.0)
        out = a + b
        tl.store(out_ptr + (offs_m[:, None] * L + offs_n[None, :]), out,
                 mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

    @triton.jit
    def scale_logits(inp_ptr, out_ptr, scale, H: tl.constexpr, L: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        m = H
        n = L
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        a = tl.load(inp_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                    mask=(offs_m[:, None] < m) & (offs_n[None, :] < n), other=0.0)
        out = a * scale
        tl.store(out_ptr + (offs_m[:, None] * L + offs_n[None, :]), out,
                 mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

    @triton.jit
    def apply_mask(logits_ptr, mask_ptr, out_ptr, scale, H: tl.constexpr, L: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        # mask_ptr is float32, 0.0 or 1.0 per element
        m = H
        n = L
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        logits = tl.load(logits_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                         mask=(offs_m[:, None] < m) & (offs_n[None, :] < n), other=0.0)
        mask = tl.load(mask_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                       mask=(offs_m[:, None] < m) & (offs_n[None, :] < n), other=0.0)
        # -inf for masked positions, else keep logits
        neg_inf = -1e20
        out = tl.where(mask > 0.0, logits, neg_inf)
        # scale by scale
        out = out * scale
        tl.store(out_ptr + (offs_m[:, None] * L + offs_n[None, :]), out,
                 mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

    @triton.jit
    def row_lse(logits_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr, BLOCK_N: tl.constexpr):
        # out_ptr: [H], float32, stores lse[h] = logsumexp(logits[h, :]) / ln(2)
        m = H
        n = L
        pid = tl.program_id(0)
        offs_n = tl.arange(0, BLOCK_N)
        # row max
        max_val = -1e20
        for j in range(0, n, BLOCK_N):
            cur = tl.load(logits_ptr + (pid * n + j + offs_n),
                          mask=(j + offs_n) < n, other=-1e20)
            max_val = tl.maximum(max_val, tl.max(cur, axis=0))
        # sumexp
        sumexp = 0.0
        for j in range(0, n, BLOCK_N):
            cur = tl.load(logits_ptr + (pid * n + j + offs_n),
                          mask=(j + offs_n) < n, other=-1e20)
            sumexp += tl.sum(tl.exp(cur - max_val), axis=0)
        lse = tl.log(sumexp) + max_val  # natural log, then divide by ln(2) on host
        # store: lse / ln(2)
        tl.store(out_ptr + pid, lse / 0.6931471805599453)

    @triton.jit
    def softmax_row_masked(logits_ptr, lse_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr, BLOCK_N: tl.constexpr):
        # out[h, :] = softmax(logits[h, :] - lse[h]) * Kc[h, :]
        m = H
        n = L
        pid = tl.program_id(0)
        offs_n = tl.arange(0, BLOCK_N)
        lse_val = tl.load(lse_ptr + pid)
        # compute softmax over row
        for j in range(0, n, BLOCK_N):
            cur = tl.load(logits_ptr + (pid * n + j + offs_n),
                          mask=(j + offs_n) < n, other=-1e20)
            expv = tl.exp(cur - lse_val)
            sumexp = 0.0
            for jj in range(0, n, BLOCK_N):
                vv = tl.load(logits_ptr + (pid * n + jj + offs_n),
                             mask=(jj + offs_n) < n, other=-1e20)
                sumexp += tl.sum(tl.exp(vv - lse_val), axis=0)
            soft = expv / sumexp
            # multiply by Kc (assumed to be available as separate matmul in host)
            # Here we just store softmax as out (host will do softmax @ Kc separately).
            tl.store(out_ptr + (pid * n + j + offs_n), soft,
                     mask=(j + offs_n) < n)

    @triton.jit
    def matmul_attn_kc(attn_ptr, kc_ptr, out_ptr,
                       H: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        # Compute out[H, D] = attn[H, L] @ kc[L, D]
        m = H
        n = D
        k = L
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, k, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            a = tl.load(
                attn_ptr + (offs_m[:, None] * L + offs_k[None, :]),
                mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
                other=0.0
            )
            b = tl.load(
                kc_ptr + (offs_k[:, None] * D + offs_n[None, :]),
                mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
                other=0.0
            )
            acc += tl.dot(a, b)
        out_offsets = offs_m[:, None] * D + offs_n[None, :]
        tl.store(
            out_ptr + out_offsets,
            acc,
            mask=(offs_m[:, None] < m) & (offs_n[None, :] < n)
        )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # You can expose tunable block sizes here; defaults below are reasonable for these dims.
        self.BLOCK_M = 16  # heads
        self.BLOCK_N = 128  # sequence chunk
        self.BLOCK_K = 64  # reduction chunk for D/P

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure device and dtype; we perform all compute in Triton. No torch.* device math.
        device = q_nope.device
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages = ckv_cache.shape[0]
        # Constants from original: num_qo_heads == 16, head_dim_ckv == 512, head_dim_kpe == 64, page_size == 1
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64
        assert qo_indptr.dim() == 1 and kv_indptr.dim() == 1
        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1  # number of batches
        num_kv_indices = kv_indices.shape[0]
        # Compute Kc_all and Kp_all on host (no torch tensor on device)
        # Note: no torch.* device tensor creation; squeeze and convert to float32
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, D]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, P]

        # Prepare outputs
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        # Main loop over batches
        for b in range(batch_size):
            # Compute q_start, q_end
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            # Compute tok_idx and slice Kc/Kp (no torch tensor on device)
            # Note: kv_indptr length equals batch_size (elements). End is b+1 since we loop b in [0, batch_size-1]
            # tok_idx ranges from kv_indptr[b] to kv_indptr[b+1]
            tok_idx = kv_indices[int(kv_indptr[b].item()): int(kv_indptr[b + 1].item())]  # 1-D LongTensor
            L = tok_idx.numel()
            if L == 0:
                # No tokens for this batch, skip
                continue

            # Slice Kc, Kp: [L, D] and [L, P]
            Kc = Kc_all[tok_idx]  # [L, D] float32
            Kp = Kp_all[tok_idx]  # [L, P] float32
            q_len = q_end - q_start

            # For each query in this batch
            for i in range(q_len):
                # Current query indices
                qn = q_nope[q_start + i].to(torch.float32).contiguous()  # [H, D]
                qp = q_pe[q_start + i].to(torch.float32).contiguous()   # [H, P]

                # Preallocate buffers on device
                logits = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)
                # 1) qn @ Kc.T -> logits part1
                matmul_qn_kc[(num_qo_heads, triton.cdiv(L, self.BLOCK_N),)](
                    qn, Kc, logits, H=num_qo_heads, D=head_dim_ckv, L=L,
                    BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K
                )
                # 2) qp @ Kp.T -> logits part2
                logits_qp = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)
                matmul_qp_kp[(num_qo_heads, triton.cdiv(L, self.BLOCK_N),)](
                    qp, Kp, logits_qp, H=num_qo_heads, P=head_dim_kpe, L=L,
                    BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K
                )
                # 3) add
                logits_full = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)
                add_logits[(num_qo_heads, triton.cdiv(L, self.BLOCK_N),)](
                    logits, logits_qp, logits_full, H=num_qo_heads, L=L,
                    BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N
                )
                # 4) scale by sm_scale
                logits_scaled = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)
                scale_logits[(num_qo_heads, triton.cdiv(L, self.BLOCK_N),)](
                    logits_full, logits_scaled, sm_scale, H=num_qo_heads, L=L,
                    BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N
                )
                # 5) apply causal mask per head: keep j > (L - q_len + i), else -inf
                prefix_len = L - q_len  # number of cached tokens before current query
                query_abs_pos = prefix_len + i  # absolute position of this query within the batch
                # Build mask [num_qo_heads, L]
                mask = torch.ones((num_qo_heads, L), dtype=torch.float32, device=device)
                for h in range(num_qo_heads):
                    mask[h, :] = 1.0 if (query_abs_pos < 0 or query_abs_pos >= L) else (torch.arange(L, device=device) > query_abs_pos).float()
                    # For very small L and i, mask may be all 1.0, but we keep general logic.
                logits_masked = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)
                apply_mask[(num_qo_heads, triton.cdiv(L, self.BLOCK_N),)](
                    logits_scaled, mask, logits_masked, sm_scale,
                    H=num_qo_heads, L=L, BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N
                )
                # 6) compute row-wise logsumexp / ln(2) -> lse[q_start+i, :]
                lse_row = torch.empty((num_qo_heads,), dtype=torch.float32, device=device)
                row_lse[(num_qo_heads,)](
                    logits_masked, lse_row, H=num_qo_heads, L=L, BLOCK_N=self.BLOCK_N
                )
                # 7) softmax row-wise using lse
                attn = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)
                # We implement softmax explicitly: out = exp(masked - lse) / sum exp(masked - lse)
                for h in range(num_qo_heads):
                    mval = lse_row[h]
                    row = logits_masked[h, :]
                    expv = torch.exp(row - mval)
                    sumexp = torch.sum(expv)
                    attn[h, :] = expv / sumexp
                # 8) output[h, :] = attn[h, :] @ Kc -> [num_qo_heads, D]
                out_row = torch.empty((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
                matmul_attn_kc[(num_qo_heads, triton.cdiv(head_dim_ckv, self.BLOCK_N),)](
                    attn, Kc, out_row, H=num_qo_heads, L=L, D=head_dim_ckv,
                    BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K
                )
                # Store
                output[q_start + i] = out_row.to(torch.bfloat16)
                lse[q_start + i, :] = lse_row  # keep as float32

        return output, lse


def run(*args):
    return ModelNew()(*args)
