import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernels
if TRITON_AVAILABLE:
    # GEMM: out[H, L] = qn[H, D] @ kc[L, D]^T
    @triton.jit
    def matmul_qn_kc(qn_ptr, kc_ptr, out_ptr,
                     H: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
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
            # qn tile: [BLOCK_M, BLOCK_K]
            q = tl.load(
                qn_ptr + (offs_m[:, None] * D + offs_k[None, :]),
                mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
                other=0.0
            )
            # kc tile: kc[n, k] -> [offs_k, offs_n]
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


    # GEMM: out[H, L] = qp[H, P] @ kp[L, P]^T
    @triton.jit
    def matmul_qp_kp(qp_ptr, kp_ptr, out_ptr,
                     H: tl.constexpr, P: tl.constexpr, L: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
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


    # Elementwise add: C = A + B
    @triton.jit
    def add(a_ptr, b_ptr, out_ptr, size: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * size + tl.arange(0, size)
        a = tl.load(a_ptr + offs)
        b = tl.load(b_ptr + offs)
        c = a + b
        tl.store(out_ptr + offs, c)


    # Elementwise scale: out = A * scale
    @triton.jit
    def scale(a_ptr, out_ptr, size: tl.constexpr, scale: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * size + tl.arange(0, size)
        a = tl.load(a_ptr + offs)
        out = a * scale
        tl.store(out_ptr + offs, out)


    # Apply causal mask: out[h, j] = logits[h, j] if j > query_abs_pos else -inf
    @triton.jit
    def apply_mask(logits_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr, query_abs_pos: tl.constexpr):
        # Each row is processed by a program id
        pid = tl.program_id(0)
        row = pid  # rows are 0..H-1
        if row >= H:
            return
        # For each column j, apply mask
        for j in range(0, L):
            val = tl.load(logits_ptr + row * L + j)
            if j <= query_abs_pos:
                val = -float("inf")
            tl.store(out_ptr + row * L + j, val)


    # Compute row-wise logsumexp in natural log: lse[h] = log(sum_j exp(logits[h, j] - max)) + max
    # Two passes: first to get max, second to get sumexp. We'll allocate lse and fill it.
    @triton.jit
    def row_logsumexp_masked(masked_ptr, lse_ptr, H: tl.constexpr, L: tl.constexpr):
        pid = tl.program_id(0)
        h = pid
        if h >= H:
            return
        # Pass 1: max
        max_val = -float("inf")
        for j in range(0, L):
            val = tl.load(masked_ptr + h * L + j)
            if val > max_val:
                max_val = val
        # Pass 2: sumexp
        sumexp = 0.0
        for j in range(0, L):
            val = tl.load(masked_ptr + h * L + j)
            sumexp += tl.exp(val - max_val)
        # Store lse in natural log
        tl.store(lse_ptr + h, tl.log(sumexp) + max_val)


    # Row-wise softmax using lse[h]: softmax[h, j] = exp(masked[h, j] - lse[h]) / denom
    @triton.jit
    def row_softmax(masked_ptr, lse_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr):
        pid = tl.program_id(0)
        h = pid
        if h >= H:
            return
        lse_val = tl.load(lse_ptr + h)
        denom = 0.0
        for j in range(0, L):
            val = tl.load(masked_ptr + h * L + j)
            denom += tl.exp(val - lse_val)
        # Normalize and write
        for j in range(0, L):
            val = tl.load(masked_ptr + h * L + j)
            soft = tl.exp(val - lse_val) / denom
            tl.store(out_ptr + h * L + j, soft)


    # Final GEMM: out_row[H, D] = softmax[H, L] @ Kc[L, D]
    @triton.jit
    def matmul_attn_kc(softmax_ptr, kc_ptr, out_ptr,
                       H: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
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
            # softmax tile: [BLOCK_M, BLOCK_K]
            s = tl.load(
                softmax_ptr + (offs_m[:, None] * L + offs_k[None, :]),
                mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
                other=0.0
            )
            # kc tile: kc[k, n] -> [offs_k, offs_n]
            kc_tile = tl.load(
                kc_ptr + (offs_n[None, :] * D + offs_k[:, None]),
                mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
                other=0.0
            )
            acc += tl.dot(s, kc_tile)

        out_offsets = offs_m[:, None] * D + offs_n[None, :]
        tl.store(
            out_ptr + out_offsets,
            acc,
            mask=(offs_m[:, None] < m) & (offs_n[None, :] < n)
        )


def run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    # Prepare data on device (float32 for compute)
    device = q_nope.device
    total_q = int(qo_indptr.shape[0] - 1)
    H = 16
    D = 512
    P = 64
    L = int(kv_indptr[1].item()) - int(kv_indptr[0].item())  # this is global L across all batches; per batch we compute b-specific L
    # We will recompute per-batch L inside loops; this is fine.

    # Precompute Kc_all and Kp_all by squeezing caches (asserting fixed shapes)
    Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, D]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, P]

    output = torch.zeros(
        (total_q, H, D), dtype=torch.bfloat16, device=device
    )
    lse = torch.full(
        (total_q, H), -float("inf"), dtype=torch.float32, device=device
    )

    # Process per batch
    for b in range(int(kv_indptr.shape[0]) - 1):  # assuming batch size equals number of kv_indptr entries - 1
        # Compute per-batch lengths and start/end
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        Q = q_end - q_start

        # For kv_indices: since per-batch L may differ, compute L and tok_idx
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        Lb = max(0, page_end - page_beg)
        if Lb == 0:
            continue
        tok_idx = kv_indices[page_beg:page_end].to(torch.long)

        # Slice Kc and Kp (host-side indexing for metadata; caches are static, so this is fine)
        Kc = Kc_all[tok_idx]  # [Lb, D]
        Kp = Kp_all[tok_idx]  # [Lb, P]

        # Loop over queries in this batch
        for i in range(Q):
            qn = q_nope[q_start + i].to(torch.float32).contiguous()  # [H, D]
            qp = q_pe[q_start + i].to(torch.float32).contiguous()   # [H, P]

            # Compute logits_qn = qn @ Kc.T -> [H, Lb]
            logits_qn = torch.empty((H, Lb), dtype=torch.float32, device=device)
            BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 128
            grid_qn = (triton.cdiv(H, BLOCK_M), triton.cdiv(Lb, BLOCK_N))
            matmul_qn_kc[grid_qn](qn, Kc, logits_qn, H, D, Lb, BLOCK_M, BLOCK_N, BLOCK_K)

            # Compute logits_qp = qp @ Kp.T -> [H, Lb]
            logits_qp = torch.empty((H, Lb), dtype=torch.float32, device=device)
            grid_qp = (triton.cdiv(H, BLOCK_M), triton.cdiv(Lb, BLOCK_N))
            matmul_qp_kp[grid_qp](qp, Kp.transpose(0, 1).contiguous(), logits_qp, H, P, Lb, BLOCK_M, BLOCK_N, BLOCK_K)

            # Add and scale
            logits = logits_qn + logits_qp  # [H, Lb]
            logits_scaled = torch.empty_like(logits)
            size = H * Lb
            grid_add = (triton.cdiv(size, 1024),)
            scale_kernel_scale = float(sm_scale)
            scale[grid_add](logits, logits_scaled, size, scale_kernel_scale)

            # Apply causal mask: j <= (Lb - Q + i) -> -inf
            query_abs_pos = (Lb - Q) + i
            masked = torch.empty_like(logits_scaled)
            grid_mask = (H,)
            apply_mask[grid_mask](logits_scaled, masked, H, Lb, query_abs_pos)

            # Compute lse per row
            lse_row = torch.empty((H,), dtype=torch.float32, device=device)
            grid_lse = (H,)
            row_logsumexp_masked[grid_lse](masked, lse_row, H, Lb)

            # Softmax per row
            softmax_row = torch.empty((H, Lb), dtype=torch.float32, device=device)
            grid_softmax = (H,)
            row_softmax[grid_softmax](masked, lse_row, softmax_row, H, Lb)

            # Final output: softmax_row @ Kc -> [H, D]
            out_row = torch.empty((H, D), dtype=torch.float32, device=device)
            BLOCK_M2, BLOCK_N2, BLOCK_K2 = 64, 64, 128
            grid_attn = (triton.cdiv(H, BLOCK_M2), triton.cdiv(D, BLOCK_N2))
            matmul_attn_kc[grid_attn](softmax_row, Kc, out_row, H, Lb, D, BLOCK_M2, BLOCK_N2, BLOCK_K2)

            # Store output in bfloat16
            output[q_start + i] = out_row.to(torch.bfloat16)

    return output, lse


def get_inputs():
    # Keep the same inputs signature as original
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to('cuda')
    _n = 1; _t = 34
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to('cuda')
    kv_indices = torch.randint(0, 989669, [34], dtype=torch.int32, device='cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale]


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure everything is on the same device
        device = q_nope.device
        # Triton kernels require CUDA; if not available, fallback (but Triton is available here)
        output, lse = run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale)
        return output, lse


def run(*args):
    return ModelNew()(*args)
