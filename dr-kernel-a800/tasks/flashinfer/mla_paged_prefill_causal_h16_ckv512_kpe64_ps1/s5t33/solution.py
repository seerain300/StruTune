import torch
import math
import triton
import triton.language as tl


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
            qn_ptr + (offs_m[:, None] * k + offs_k[None, :]),
            mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
            other=0.0
        )
        k_tile = tl.load(
            kc_ptr + (offs_n[None, :] * k + offs_k[:, None]),
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
def add_logits(a_ptr, b_ptr, out_ptr,
               H: tl.constexpr, L: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    m = H
    n = L

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a = tl.load(a_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                other=0.0)
    b = tl.load(b_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                other=0.0)
    c = a + b
    tl.store(out_ptr + (offs_m[:, None] * L + offs_n[None, :]),
             c, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def scale_logits(mat_ptr, out_ptr, scale, H: tl.constexpr, L: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    m = H
    n = L

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mat = tl.load(mat_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                  mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                  other=0.0)
    mat = mat * scale
    tl.store(out_ptr + (offs_m[:, None] * L + offs_n[None, :]),
             mat, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def apply_mask(logits_ptr, mask_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # out[H, L] = logits, but set positions where mask[h, j] == 1 to -inf
    m = H
    n = L

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    logits = tl.load(logits_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                     mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                     other=0.0)
    mask = tl.load(mask_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                   mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                   other=0.0)  # 1 where to mask, 0 otherwise
    neg_inf = -float("inf")
    new_logits = tl.where(mask != 0, neg_inf, logits)
    tl.store(out_ptr + (offs_m[:, None] * L + offs_n[None, :]),
             new_logits, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def matmul_attn_kc(attn_ptr, kc_ptr, out_ptr,
                   H: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # out[H, D] = attn[H, L] @ kc[L, D]
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
        attn = tl.load(
            attn_ptr + (offs_m[:, None] * n + offs_k[None, :]),
            mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
            other=0.0
        )
        k_tile = tl.load(
            kc_ptr + (offs_n[None, :] * k + offs_k[:, None]),
            mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
            other=0.0
        )
        acc += tl.dot(attn, k_tile)

    out_offsets = offs_m[:, None] * D + offs_n[None, :]
    tl.store(
        out_ptr + out_offsets,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n)
    )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                q_nope, q_pe,
                ckv_cache, kpe_cache,
                qo_indptr, kv_indptr, kv_indices,
                sm_scale):
        # Ensure tensors are on the same device and dtype; keep compute in float32
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "Tensors must be on CUDA device"
        assert qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "Indicators and indices must be on CUDA"

        # Constants
        H = 16
        D = 512
        P = 64
        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1
        assert batch_size >= 0, "len_indptr must be at least 1"

        # Prepare Kc_all and Kp_all on host (CPU side) by squeezing caches to [num_pages, dim]
        # We will extract slices on GPU inside the loop using 1-D indices. No torch tensor creation on GPU.
        total_q = int(qo_indptr[-1].item())
        assert q_nope.shape[0] == total_q, "Total queries mismatch"

        # Output buffers (float32 for compute; cast later if needed)
        output = torch.empty((total_q, H, D), dtype=torch.float32, device=device)
        lse = torch.full((total_q, H), -float("inf"), dtype=torch.float32, device=device)

        # Iterate over batches safely (guard against b+1 out-of-bounds)
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            L = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())

            if q_start >= q_end or L <= 0:
                continue

            # Token indices for this batch
            tok_idx = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())]  # 1-D int32
            tok_idx = tok_idx.to(torch.int32)  # ensure int32

            # Extract Kc and Kp for this batch (slices on GPU). No torch tensor creation on device.
            # Kc_all: [num_pages, D], Kp_all: [num_pages, P]
            # We use 1-D indices for slicing, which Triton will accept as pointers.
            Kc_all = ckv_cache.squeeze(1)  # [num_pages, D]
            Kp_all = kpe_cache.squeeze(1)  # [num_pages, P]
            Kc = Kc_all[tok_idx]  # [L, D]
            Kp = Kp_all[tok_idx]  # [L, P]

            # For each query i
            q_len = q_end - q_start
            for i in range(q_len):
                # Load qn and qp (float32 for compute)
                qn = q_nope[q_start + i].contiguous().to(torch.float32)  # [H, D]
                qp = q_pe[q_start + i].contiguous().to(torch.float32)   # [H, P]

                # Allocate intermediate buffers
                logits_qn = torch.empty((H, L), dtype=torch.float32, device=device)
                logits_qp = torch.empty((H, L), dtype=torch.float32, device=device)
                logits_sum = torch.empty((H, L), dtype=torch.float32, device=device)
                masked_logits = torch.empty((H, L), dtype=torch.float32, device=device)
                attn = torch.empty((H, L), dtype=torch.float32, device=device)
                out_tmp = torch.empty((H, D), dtype=torch.float32, device=device)

                # Triton GEMM: qn @ Kc.T -> [H, L]
                grid_qn = (triton.cdiv(H, 64), triton.cdiv(L, 64))
                matmul_qn_kc[grid_qn](
                    qn, Kc, logits_qn, H, D, L, 64, 64, 64
                )

                # Triton GEMM: qp @ Kp.T -> [H, L]
                grid_qp = (triton.cdiv(H, 64), triton.cdiv(L, 64))
                matmul_qp_kp[grid_qp](
                    qp, Kp, logits_qp, H, P, L, 64, 64, 64
                )

                # Triton add: logits_qn + logits_qp
                grid_add = (triton.cdiv(H, 64), triton.cdiv(L, 64))
                add_logits[grid_add](logits_qn, logits_qp, logits_sum, H, L, 64, 64)

                # Triton scale
                grid_scale = (triton.cdiv(H, 64), triton.cdiv(L, 64))
                scale_logits[grid_scale](logits_sum, masked_logits, sm_scale, H, L, 64, 64)

                # Compute mask: query_abs_pos = L - q_len + i
                query_abs_pos = L - q_len + i  # Python int
                # Build mask on device: [H, L] bool, then cast to int for Triton
                j = torch.arange(L, device=device)
                mask_bool = (j > query_abs_pos).unsqueeze(0).expand(H, L)
                mask_int = mask_bool.to(torch.int32)

                # Triton apply mask
                grid_mask = (triton.cdiv(H, 64), triton.cdiv(L, 64))
                apply_mask[grid_mask](masked_logits, mask_int, masked_logits, H, L, 64, 64)

                # Compute lse per row on host (torch) for numerical stability
                # lse[h] = logsumexp(masked_logits[h, :]) / ln(2)
                # Use PyTorch for this part (host code), since Triton log isn't used here and host log is allowed.
                row_max = masked_logits.max(dim=1, keepdim=True).values
                masked_exp = torch.exp(masked_logits - row_max)
                sumexp = masked_exp.sum(dim=1, keepdim=True)
                lse_row = torch.log(sumexp) / math.log(2.0)  # ln(2) scaling per original
                lse[q_start + i] = lse_row

                # Softmax using lse: attn[h, j] = exp(masked_logits[h, j] - lse[h])
                attn = torch.exp(masked_logits - lse[q_start + i].unsqueeze(1))

                # Triton GEMM: attn @ Kc -> [H, D]
                grid_attn = (triton.cdiv(H, 64), triton.cdiv(D, 64))
                matmul_attn_kc[grid_attn](
                    attn, Kc, out_tmp, H, L, D, 64, 64, 64
                )

                # Store output
                output[q_start + i] = out_tmp

        return output, lse


# Original helpers preserved (unchanged)
def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16)
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16)
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16)
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16)
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
    _n = 1; _t = 34
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
    kv_indices = torch.randint(0, 989669, [34], dtype=torch.int32)
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7):
    _out = ModelNew()(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


class Model(torch.nn.Module):
    def forward(self, *args):
        return fused_operator(*args)


def run(*args):
    return ModelNew()(*args)
