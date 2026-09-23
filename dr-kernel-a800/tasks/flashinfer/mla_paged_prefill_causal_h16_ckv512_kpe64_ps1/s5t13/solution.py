import torch
import triton
import triton.language as tl
import math

# Triton kernels
@triton.jit
def matmul_qn_kc(qn_ptr, kc_ptr, out_ptr,
                 H: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # out[h, l] = sum_d qn[h, d] * kc[l, d]
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
    # out[h, l] = sum_p qp[h, p] * kp[l, p]
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
def add_logits(a_ptr, b_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr):
    m = H
    n = L
    for h in range(0, m):
        for j in range(0, n):
            a = tl.load(a_ptr + h * n + j)
            b = tl.load(b_ptr + h * n + j)
            tl.store(out_ptr + h * n + j, a + b)


@triton.jit
def scale_logits(in_ptr, out_ptr, scale, H: tl.constexpr, L: tl.constexpr):
    m = H
    n = L
    for h in range(0, m):
        for j in range(0, n):
            v = tl.load(in_ptr + h * n + j)
            tl.store(out_ptr + h * n + j, v * scale)


@triton.jit
def apply_mask(logits_ptr, masked_ptr, lse_ptr, pos, H: tl.constexpr, L: tl.constexpr, scale: tl.float32):
    # mask per row h: j > pos -> keep, else set to -inf; then shift by lse[h] * scale (scale=1.0 here)
    m = H
    n = L
    for h in range(0, m):
        maxv = -float("inf")
        # first pass: find max
        for j in range(0, n):
            v = tl.load(logits_ptr + h * n + j)
            if j > pos:
                maxv = tl.maximum(maxv, v)
        # compute sumexp with shifted -inf
        sumexp = 0.0
        for j in range(0, n):
            v = tl.load(logits_ptr + h * n + j)
            keep = j > pos
            # masked value: keep v, else -inf
            vm = tl.where(keep, v, -float("inf"))
            sumexp += tl.exp(vm - maxv)
        # compute lse and store
        lse_val = tl.log(sumexp) + maxv  # logsumexp with -inf masked
        # store lse for this row (lse in float32 buffer)
        tl.store(lse_ptr + h, lse_val)
        # write masked logits shifted by lse
        for j in range(0, n):
            v = tl.load(logits_ptr + h * n + j)
            keep = j > pos
            vm = tl.where(keep, v, -float("inf"))
            tl.store(masked_ptr + h * n + j, vm - lse_val)


@triton.jit
def softmax_row_masked(logits_ptr, out_ptr, lse_ptr, H: tl.constexpr, L: tl.constexpr):
    m = H
    n = L
    # this kernel assumes logits_ptr points to already masked and shifted values
    for h in range(0, m):
        lse = tl.load(lse_ptr + h)
        sumexp = 0.0
        for j in range(0, n):
            v = tl.load(logits_ptr + h * n + j)
            sumexp += tl.exp(v - lse)
        inv_sum = 1.0 / sumexp
        for j in range(0, n):
            v = tl.load(logits_ptr + h * n + j)
            p = tl.exp(v - lse) * inv_sum
            tl.store(out_ptr + h * n + j, p)


@triton.jit
def matmul_attn_kc(attn_ptr, kc_ptr, out_ptr,
                   H: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # out[h, d] = sum_l attn[h, l] * kc[l, d]
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
        k_tile = tl.load(
            kc_ptr + (offs_k[:, None] * D + offs_n[None, :]),
            mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
            other=0.0
        )
        acc += tl.dot(a, k_tile)

    out_offsets = offs_m[:, None] * D + offs_n[None, :]
    tl.store(
        out_ptr + out_offsets,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n)
    )


# ModelNew: Triton-only forward
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # All heavy computation in Triton kernels; avoid any torch tensor math on device
        device = q_nope.device
        dtype_compute = torch.float32

        # Prepare Kc_all and Kp_all (squeeze to 2-D [num_pages, D] and [num_pages, P])
        Kc_all = ckv_cache.squeeze(1).to(dtype=dtype_compute).contiguous()  # [num_pages, D]
        Kp_all = kpe_cache.squeeze(1).to(dtype=dtype_compute).contiguous()  # [num_pages, P]

        # Prepare outputs
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        assert num_qo_heads == 16, "num_qo_heads must be 16"
        assert head_dim_ckv == 512, "head_dim_ckv must be 512"
        assert head_dim_kpe == 64, "head_dim_kpe must be 64"

        output = torch.empty(
            (total_q, num_qo_heads, head_dim_ckv),
            dtype=torch.bfloat16,
            device=device
        )
        lse = torch.empty(
            (total_q, num_qo_heads),
            dtype=torch.float32,
            device=device
        )

        # Iterate over batches safely: b < len(qo_indptr) - 1
        len_qo = qo_indptr.shape[0]
        for b in range(0, len_qo - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())

            if q_start >= q_end:
                continue

            # token indices for this batch
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                continue

            tok_idx = kv_indices[page_beg:page_end].to(torch.int32).contiguous()  # [L]
            L = int(page_end - page_beg)
            D = int(head_dim_ckv)
            P = int(head_dim_kpe)
            H = int(num_qo_heads)

            # Slice Kc and Kp for this batch
            Kc = Kc_all[tok_idx]  # [L, D]
            Kp = Kp_all[tok_idx]  # [L, P]

            # Batch loop: each query i
            for i in range(q_start, q_end):
                # Load qn and qp
                qn = q_nope[i].to(dtype=dtype_compute).contiguous()  # [H, D]
                qp = q_pe[i].to(dtype=dtype_compute).contiguous()   # [H, P]

                # Matmuls
                logits_qn = torch.empty((H, L), dtype=torch.float32, device=device)
                logits_qp = torch.empty((H, L), dtype=torch.float32, device=device)

                # Grids
                BLOCK_M = 16
                BLOCK_N = 64
                BLOCK_K = 64

                grid_qn = (H + BLOCK_M - 1) // BLOCK_M, (L + BLOCK_N - 1) // BLOCK_N
                grid_qp = (H + BLOCK_M - 1) // BLOCK_M, (L + BLOCK_N - 1) // BLOCK_N

                matmul_qn_kc[grid_qn](
                    qn, Kc, logits_qn,
                    H, D, L,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
                )

                matmul_qp_kp[grid_qp](
                    qp, Kp, logits_qp,
                    H, P, L,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
                )

                # Add, scale
                logits_sum = torch.empty((H, L), dtype=torch.float32, device=device)
                add_logits[(H, L)](logits_qn, logits_qp, logits_sum, H, L)
                scale_logits[(H, L)](logits_sum, logits_sum, sm_scale, H, L)

                # Mask per row h: j > (L - (q_end - q_start) + i)
                Q_batch = q_end - q_start
                # Compute absolute position: prefix_len = (q_start - q_start) + (q_end - q_start) - Q_batch = q_end - q_start - Q_batch
                prefix_len = (q_end - q_start) - Q_batch
                pos = prefix_len + i  # absolute query position

                # Apply mask and compute lse
                masked = torch.empty((H, L), dtype=torch.float32, device=device)
                # We need to pass H, L; Triton expects constexpr, so we call with compile-time H,L values if available
                # Here H,L are runtime ints; Triton kernels above are annotated with tl.constexpr on H/L, but we pass as args.
                apply_mask[(H, L)](logits_sum, masked, lse[i], pos, H, L, 1.0)  # sm_scale is 1.0 here; lse buffer is 1D

                # Softmax on masked logits shifted by lse
                attn = torch.empty((H, L), dtype=torch.float32, device=device)
                softmax_row_masked[(H, L)](masked, attn, lse[i], H, L)

                # Output = attn @ Kc
                out_h = torch.empty((H, D), dtype=torch.float32, device=device)
                matmul_attn_kc[(H, (D + BLOCK_N - 1) // BLOCK_N, L + (BLOCK_K - 1) // BLOCK_K)](
                    attn, Kc, out_h,
                    H, L, D,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
                )

                # Store output in bfloat16
                output[i] = out_h.to(torch.bfloat16)

        return output, lse


# Example helpers from the original snippet (not required by evaluator but provided for consistency)
def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    # qo_indptr: [len_indptr] on device
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device='cuda')
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), torch.cumsum(_lens, 0)], 0)
    # kv_indptr: [len_indptr] on device
    _n = 1; _t = 34
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device='cuda')
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), torch.cumsum(_lens, 0)], 0)
    kv_indices = torch.randint(0, 989669, [34], dtype=torch.int32, device='cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7):
    _out = ModelNew()(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


# Entry point for evaluator
class Model(torch.nn.Module):
    def forward(self, *args):
        return fused_operator(*args)


def run(*args):
    return ModelNew()(*args)
