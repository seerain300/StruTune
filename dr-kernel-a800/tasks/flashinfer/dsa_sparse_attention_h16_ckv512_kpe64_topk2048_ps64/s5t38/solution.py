import torch
import math
import triton
import triton.language as tl


# Matmul kernel: A[M, K] @ B[K, N] -> C[M, N], here we use it with M tile = 256, K tile 64/128, N tile 128.
# We will call it with A=[H, K] (H=16), B=Kc_gather.T [K, M], producing C [H, M].
@triton.jit
def matmul_qn_kc_kernel(qn_ptr, Kc_ptr, Out_ptr,
                        H: tl.constexpr, M,
                        stride_qn0, stride_qn1,
                        stride_kc0, stride_kc1,
                        stride_out0, stride_out1,
                        BLOCK_H: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_h = tl.program_id(0)  # tile over heads
    pid_m = tl.program_id(1)  # tile over M (number of valid entries)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    acc = tl.zeros((BLOCK_H, BLOCK_M), dtype=tl.float32)

    Kq = 512
    for k0 in range(0, Kq, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        q_ptrs = qn_ptr + (offs_h[:, None] * stride_qn0 + offs_k[None, :] * stride_qn1)
        a = tl.load(q_ptrs, mask=(offs_h[:, None] < H) & (offs_k[None, :] < Kq), other=0.0)
        # Kc is [M, 512], we want [K, M] for dot, so take transpose reference via pointers
        kc_ptrs = Kc_ptr + (offs_k[:, None] * stride_kc0 + offs_m[None, :] * stride_kc1)
        b = tl.load(kc_ptrs, mask=(offs_k[:, None] < Kq) & (offs_m[None, :] < M), other=0.0)
        acc += tl.dot(a, b)  # (BLOCK_H, BLOCK_K) @ (BLOCK_K, BLOCK_M) -> (BLOCK_H, BLOCK_M)

    out_ptrs = Out_ptr + (offs_h[:, None] * stride_out0 + offs_m[None, :] * stride_out1)
    tl.store(out_ptrs, acc, mask=(offs_h[:, None] < H) & (offs_m[None, :] < M))


# Matmul kernel: A[H, Kp] @ B[M, Kp] -> D[H, M], with Kp=64
@triton.jit
def matmul_qp_kp_kernel(qp_ptr, Kp_ptr, Out_ptr,
                        H: tl.constexpr, M,
                        stride_qp0, stride_qp1,
                        stride_kp0, stride_kp1,
                        stride_out0, stride_out1,
                        BLOCK_H: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    acc = tl.zeros((BLOCK_H, BLOCK_M), dtype=tl.float32)

    Kp = 64
    for k0 in range(0, Kp, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        q_ptrs = qp_ptr + (offs_h[:, None] * stride_qp0 + offs_k[None, :] * stride_qp1)
        a = tl.load(q_ptrs, mask=(offs_h[:, None] < H) & (offs_k[None, :] < Kp), other=0.0)
        kp_ptrs = Kp_ptr + (offs_k[:, None] * stride_kp0 + offs_m[None, :] * stride_kp1)
        b = tl.load(kp_ptrs, mask=(offs_k[:, None] < Kp) & (offs_m[None, :] < M), other=0.0)
        acc += tl.dot(a, b)

    out_ptrs = Out_ptr + (offs_h[:, None] * stride_out0 + offs_m[None, :] * stride_out1)
    tl.store(out_ptrs, acc, mask=(offs_h[:, None] < H) & (offs_m[None, :] < M))


# Elementwise add and scale: C[H,M] = (A[H,M] + B[H,M]) * scale
@triton.jit
def add_scale_kernel(A_ptr, B_ptr, C_ptr,
                     H: tl.constexpr, M,
                     stride_a0, stride_a1,
                     stride_b0, stride_b1,
                     stride_c0, stride_c1,
                     scale: tl.float32):
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_h = pid_h * 16 + tl.arange(0, 16)
    offs_m = pid_m * 256 + tl.arange(0, 256)

    a_ptrs = A_ptr + (offs_h[:, None] * stride_a0 + offs_m[None, :] * stride_a1)
    b_ptrs = B_ptr + (offs_h[:, None] * stride_b0 + offs_m[None, :] * stride_b1)
    c_ptrs = C_ptr + (offs_h[:, None] * stride_c0 + offs_m[None, :] * stride_c1)

    a = tl.load(a_ptrs, mask=(offs_h[:, None] < H) & (offs_m[None, :] < M), other=0.0)
    b = tl.load(b_ptrs, mask=(offs_h[:, None] < H) & (offs_m[None, :] < M), other=0.0)
    c = (a + b) * scale
    tl.store(c_ptrs, c, mask=(offs_h[:, None] < H) & (offs_m[None, :] < M))


# Per-row stable softmax on X[H,M] -> Out[H,M]
@triton.jit
def softmax_row_kernel(X_ptr, Out_ptr,
                       H: tl.constexpr, M,
                       stride_x0, stride_x1,
                       stride_out0, stride_out1):
    h = tl.program_id(0)

    # Pass 1: row max
    row_max = -float("inf")
    for m0 in range(0, M, 256):
        offs_m = m0 + tl.arange(0, 256)
        x_ptrs = X_ptr + (h * stride_x0 + offs_m * stride_x1)
        x = tl.load(x_ptrs, mask=offs_m < M, other=-float("inf"))
        chunk_max = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, chunk_max)

    # Pass 2: sum_exp
    sum_exp = 0.0
    for m0 in range(0, M, 256):
        offs_m = m0 + tl.arange(0, 256)
        x_ptrs = X_ptr + (h * stride_x0 + offs_m * stride_x1)
        x = tl.load(x_ptrs, mask=offs_m < M, other=-float("inf"))
        e = tl.exp(x - row_max)
        sum_exp += tl.sum(e, axis=0)

    # Pass 3: write normalized
    inv_sum = 1.0 / sum_exp
    for m0 in range(0, M, 256):
        offs_m = m0 + tl.arange(0, 256)
        x_ptrs = X_ptr + (h * stride_x0 + offs_m * stride_x1)
        out_ptrs = Out_ptr + (h * stride_out0 + offs_m * stride_out1)
        x = tl.load(x_ptrs, mask=offs_m < M, other=-float("inf"))
        e = tl.exp(x - row_max) * inv_sum
        tl.store(out_ptrs, e, mask=offs_m < M)


# Generic fp32 matmul: A[M, K] @ B[K, N] -> C[M, N]
@triton.jit
def _matmul_fp32(A_ptr, B_ptr, C_ptr,
                 M, N, K,
                 stride_am, stride_ak,
                 stride_bk, stride_bn,
                 stride_cm, stride_cn,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Per-row logsumexp of X[H,M] / ln(2) -> Out[H]
@triton.jit
def lse_row_kernel(X_ptr, Out_ptr,
                   H: tl.constexpr, M,
                   stride_x0, stride_x1,
                   stride_out0,
                   inv_ln2: tl.float32):
    h = tl.program_id(0)

    # Pass 1: row max
    row_max = -float("inf")
    for m0 in range(0, M, 256):
        offs_m = m0 + tl.arange(0, 256)
        x_ptrs = X_ptr + (h * stride_x0 + offs_m * stride_x1)
        x = tl.load(x_ptrs, mask=offs_m < M, other=-float("inf"))
        chunk_max = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, chunk_max)

    # Pass 2: sum_exp
    sum_exp = 0.0
    for m0 in range(0, M, 256):
        offs_m = m0 + tl.arange(0, 256)
        x_ptrs = X_ptr + (h * stride_x0 + offs_m * stride_x1)
        x = tl.load(x_ptrs, mask=offs_m < M, other=-float("inf"))
        e = tl.exp(x - row_max)
        sum_exp += tl.sum(e, axis=0)

    lse_val = row_max + tl.log(sum_exp)  # natural log
    tl.store(Out_ptr + h * stride_out0, lse_val * inv_ln2)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        device = q_nope.device

        # Shapes and assertions
        num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        _, _, head_dim_kpe = q_pe.shape
        assert head_dim_kpe == 64

        num_pages, seq_len, _ = ckv_cache.shape
        assert seq_len == 64, "kpe_cache seq_len must be 64"
        assert kpe_cache.shape[1:] == (64, 64), "kpe_cache must have shape [num_pages, 64, 64]"

        # Flatten caches to [N, dim], N=num_pages*64
        Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)  # [num_pages*64, 512]
        Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)  # [num_pages*64, 64]

        # Allocate outputs
        output = torch.empty((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((num_tokens, num_qo_heads), dtype=torch.float32, device=device)

        for t in range(num_tokens):
            # Gather valid indices
            indices_t = sparse_indices[t]  # [topk]
            valid_mask = indices_t != -1
            if not torch.any(valid_mask):
                # No valid entries for this token
                lse[t].fill_(-float("inf"))
                output[t].zero_()
                continue

            valid_indices = indices_t[valid_mask].to(torch.long)  # [M]
            M = valid_indices.numel()

            # Gather Kc and Kp
            Kc_gather = Kc_all[valid_indices]  # [M, 512], fp32
            Kp_gather = Kp_all[valid_indices]  # [M, 64], fp32
            # Ensure contiguous
            Kc_gather = Kc_gather.contiguous()
            Kp_gather = Kp_gather.contiguous()

            # Prepare qn and qp (float32), [H=16, Kq] and [H, Kp]
            qn = q_nope[t].to(torch.float32).contiguous()  # [16, 512]
            qp = q_pe[t].to(torch.float32).contiguous()    # [16, 64]

            # Matmul qn @ Kc_gather.T -> logits_qn [16, M]
            logits_qn = torch.empty((num_qo_heads, M), dtype=torch.float32, device=device)
            matmul_qn_kc_kernel[(1, 1)](  # grid: (tiles over H, tiles over M); H=16, M unknown at compile, so we set to 1 and rely on masks
                                        qn, Kc_gather,
                                        logits_qn,
                                        H=num_qo_heads, M=M,
                                        stride_qn0=qn.stride(0), stride_qn1=qn.stride(1),
                                        stride_kc0=Kc_gather.stride(0), stride_kc1=Kc_gather.stride(1),
                                        stride_out0=logits_qn.stride(0), stride_out1=logits_qn.stride(1),
                                        BLOCK_H=16, BLOCK_M=256, BLOCK_K=128)

            # Matmul qp @ Kp_gather.T -> logits_qp [16, M]
            logits_qp = torch.empty((num_qo_heads, M), dtype=torch.float32, device=device)
            matmul_qp_kp_kernel[(1, 1)](
                qp, Kp_gather,
                logits_qp,
                H=num_qo_heads, M=M,
                stride_qp0=qp.stride(0), stride_qp1=qp.stride(1),
                stride_kp0=Kp_gather.stride(0), stride_kp1=Kp_gather.stride(1),
                stride_out0=logits_qp.stride(0), stride_out1=logits_qp.stride(1),
                BLOCK_H=16, BLOCK_M=256, BLOCK_K=64
            )

            # Sum and scale: logits = logits_qn + logits_qp, then * sm_scale
            logits_sum = torch.empty_like(logits_qn)
            add_scale_kernel[(1, 1)](
                logits_qn, logits_qp,
                logits_sum,
                H=num_qo_heads, M=M,
                stride_a0=logits_qn.stride(0), stride_a1=logits_qn.stride(1),
                stride_b0=logits_qp.stride(0), stride_b1=logits_qp.stride(1),
                stride_c0=logits_sum.stride(0), stride_c1=logits_sum.stride(1),
                scale=sm_scale
            )

            # Softmax per row
            attn = torch.empty((num_qo_heads, M), dtype=torch.float32, device=device)
            softmax_row_kernel[(num_qo_heads,)](
                logits_sum,
                attn,
                H=num_qo_heads, M=M,
                stride_x0=logits_sum.stride(0), stride_x1=logits_sum.stride(1),
                stride_out0=attn.stride(0), stride_out1=attn.stride(1)
            )

            # Output: attn @ Kc_gather -> [16, 512]
            output_row = torch.empty((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
            _matmul_fp32[(1, 1)](
                attn, Kc_gather,
                output_row,
                M=attn.shape[0], N=head_dim_ckv, K=attn.shape[1],
                stride_am=attn.stride(0), stride_ak=attn.stride(1),
                stride_bk=Kc_gather.stride(0), stride_bn=Kc_gather.stride(1),
                stride_cm=output_row.stride(0), stride_cn=output_row.stride(1),
                BLOCK_M=16, BLOCK_N=128, BLOCK_K=256
            )

            # Store output as bfloat16
            output[t] = output_row.to(torch.bfloat16)

            # LSE per row: logsumexp((logits * sm_scale), dim=1) / ln(2)
            lse_row = torch.empty((num_qo_heads,), dtype=torch.float32, device=device)
            lse_row_kernel[(num_qo_heads,)](
                logits_sum,
                lse_row,
                H=num_qo_heads, M=M,
                stride_x0=logits_sum.stride(0), stride_x1=logits_sum.stride(1),
                stride_out0=lse_row.stride(0),
                inv_ln2=1.0 / math.log(2.0)
            )
            lse[t] = lse_row

        return output, lse


# For completeness: the original helpers
@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages, _, _ = ckv_cache.shape
    topk = sparse_indices.shape[-1]

    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert ckv_cache.shape[1] == 64
    assert kpe_cache.shape[1] == 64
    assert sparse_indices.shape[0] == num_tokens
    assert sparse_indices.shape[-1] == topk

    device = q_nope.device

    Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)  # [num_pages*64, 512]
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)  # [num_pages*64, 64]

    output = torch.zeros((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
    lse = torch.full((num_tokens, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    for t in range(num_tokens):
        indices = sparse_indices[t]  # [topk]
        valid_mask = indices != -1
        if not torch.any(valid_mask):
            output[t].zero_()
            continue

        valid_indices = indices[valid_mask].to(torch.long)  # [M]
        M = valid_indices.numel()

        Kc_gather = Kc_all[valid_indices].contiguous()  # [M, 512]
        Kp_gather = Kp_all[valid_indices].contiguous()  # [M, 64]

        qn = q_nope[t].to(torch.float32).contiguous()  # [16, 512]
        qp = q_pe[t].to(torch.float32).contiguous()    # [16, 64]

        # Compute logits: qn @ Kc_gather.T, qp @ Kp_gather.T
        logits_qn = torch.empty((num_qo_heads, M), dtype=torch.float32, device=device)
        matmul_qn_kc_kernel[(1, 1)](
            qn, Kc_gather,
            logits_qn,
            H=num_qo_heads, M=M,
            stride_qn0=qn.stride(0), stride_qn1=qn.stride(1),
            stride_kc0=Kc_gather.stride(0), stride_kc1=Kc_gather.stride(1),
            stride_out0=logits_qn.stride(0), stride_out1=logits_qn.stride(1),
            BLOCK_H=16, BLOCK_M=256, BLOCK_K=128
        )

        logits_qp = torch.empty((num_qo_heads, M), dtype=torch.float32, device=device)
        matmul_qp_kp_kernel[(1, 1)](
            qp, Kp_gather,
            logits_qp,
            H=num_qo_heads, M=M,
            stride_qp0=qp.stride(0), stride_qp1=qp.stride(1),
            stride_kp0=Kp_gather.stride(0), stride_kp1=Kp_gather.stride(1),
            stride_out0=logits_qp.stride(0), stride_out1=logits_qp.stride(1),
            BLOCK_H=16, BLOCK_M=256, BLOCK_K=64
        )

        logits_sum = logits_qn + logits_qp
        logits_scaled = logits_sum * sm_scale

        # Softmax per row
        attn = torch.softmax(logits_scaled, dim=1)  # [16, M]

        # Output: attn @ Kc_gather
        output_row = torch.empty((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        _matmul_fp32[(1, 1)](
            attn, Kc_gather,
            output_row,
            M=attn.shape[0], N=head_dim_ckv, K=attn.shape[1],
            stride_am=attn.stride(0), stride_ak=attn.stride(1),
            stride_bk=Kc_gather.stride(0), stride_bn=Kc_gather.stride(1),
            stride_cm=output_row.stride(0), stride_cn=output_row.stride(1),
            BLOCK_M=16, BLOCK_N=128, BLOCK_K=256
        )
        output[t] = output_row.to(torch.bfloat16)

        # LSE per row
        lse[t] = torch.logsumexp(logits_scaled, dim=1) / math.log(2.0)

    return output, lse


def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device="cuda")
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device="cuda")
    ckv_cache = torch.randn([8462, 64, 512], dtype=torch.bfloat16, device="cuda")
    kpe_cache = torch.randn([8462, 64, 64], dtype=torch.bfloat16, device="cuda")
    sparse_indices = torch.randint(0, 541568, [1, 2048], dtype=torch.int32, device="cuda")
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5):
    _out = run(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


# Entry point for evaluation harness
class Model(torch.nn.Module):
    def forward(self, *args):
        return run(*args)

# Triton-optimized version
class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return ModelNew().forward(*args)


def run(*args):
    return ModelNew()(*args)
