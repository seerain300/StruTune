import torch
import math
import triton
import triton.language as tl


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

    A = tl.load(a_ptrs, mask=(offs_h[:, None] < H) & (offs_m[None, :] < M), other=0.0)
    B = tl.load(b_ptrs, mask=(offs_h[:, None] < H) & (offs_m[None, :] < M), other=0.0)
    C = (A + B) * scale
    tl.store(c_ptrs, C, mask=(offs_h[:, None] < H) & (offs_m[None, :] < M))


# Softmax per row: attn[H,M] = softmax(C[H,M], dim=1) (stable: subtract max)
@triton.jit
def softmax_row_kernel(C_ptr, attn_ptr,
                       H: tl.constexpr, M,
                       stride_c0, stride_c1,
                       stride_a0, stride_a1):
    h = tl.program_id(0)
    # Pass 1: row max
    row_max = -float("inf")
    for m0 in range(0, M, 256):
        offs_m = m0 + tl.arange(0, 256)
        c_ptrs = C_ptr + (h * stride_c0 + offs_m * stride_c1)
        x = tl.load(c_ptrs, mask=offs_m < M, other=-float("inf"))
        local_max = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, local_max)

    # Pass 2: row sum of exp(x - row_max)
    row_sum = 0.0
    for m0 in range(0, M, 256):
        offs_m = m0 + tl.arange(0, 256)
        c_ptrs = C_ptr + (h * stride_c0 + offs_m * stride_c1)
        x = tl.load(c_ptrs, mask=offs_m < M, other=-float("inf"))
        e = tl.exp(x - row_max)
        row_sum += tl.sum(e, axis=0)

    inv_row_sum = 1.0 / row_sum

    # Pass 3: write normalized outputs
    for m0 in range(0, M, 256):
        offs_m = m0 + tl.arange(0, 256)
        c_ptrs = C_ptr + (h * stride_c0 + offs_m * stride_c1)
        x = tl.load(c_ptrs, mask=offs_m < M, other=-float("inf"))
        y = tl.exp(x - row_max) * inv_row_sum
        attn_ptrs = attn_ptr + (h * stride_a0 + offs_m * stride_a1)
        tl.store(attn_ptrs, y, mask=offs_m < M)


# Per-row logsumexp over C[H,M], return lse[H] = logsumexp(C)/ln(2)
@triton.jit
def lse_row_kernel(C_ptr, lse_ptr,
                   H: tl.constexpr, M,
                   stride_c0, stride_c1):
    h = tl.program_id(0)
    # Pass 1: row max
    row_max = -float("inf")
    for m0 in range(0, M, 256):
        offs_m = m0 + tl.arange(0, 256)
        c_ptrs = C_ptr + (h * stride_c0 + offs_m * stride_c1)
        x = tl.load(c_ptrs, mask=offs_m < M, other=-float("inf"))
        local_max = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, local_max)

    # Pass 2: row sum of exp(x - row_max)
    sum_exp = 0.0
    for m0 in range(0, M, 256):
        offs_m = m0 + tl.arange(0, 256)
        c_ptrs = C_ptr + (h * stride_c0 + offs_m * stride_c1)
        x = tl.load(c_ptrs, mask=offs_m < M, other=-float("inf"))
        e = tl.exp(x - row_max)
        sum_exp += tl.sum(e, axis=0)

    lse_val = row_max + tl.log(sum_exp)  # natural log
    tl.store(lse_ptr + h, lse_val)


# Triton matmul specialized for logits: qn [H, Kq] x K_gather [M, Kq] -> out [H, M], Kq fixed
# We pass Kq as tl.constexpr to let Triton optimize loops. H is also constexpr.
@triton.jit
def matmul_qn_kg_kernel(qn_ptr, Kg_ptr, out_ptr,
                        H: tl.constexpr, M, Kq: tl.constexpr,
                        stride_q0, stride_q1,
                        stride_k0, stride_k1,
                        stride_o0, stride_o1,
                        BLOCK_H: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    acc = tl.zeros((BLOCK_H, BLOCK_M), dtype=tl.float32)

    for k0 in range(0, Kq, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        q_ptrs = qn_ptr + (offs_h[:, None] * stride_q0 + offs_k[None, :] * stride_q1)
        k_ptrs = Kg_ptr + (offs_m[None, :] * stride_k0 + offs_k[:, None] * stride_k1)
        q = tl.load(q_ptrs, mask=(offs_h[:, None] < H) & (offs_k[None, :] < Kq), other=0.0)
        k = tl.load(k_ptrs, mask=(offs_k[:, None] < Kq) & (offs_m[None, :] < M), other=0.0)
        acc += tl.dot(q, k)

    out_ptrs = out_ptr + (offs_h[:, None] * stride_o0 + offs_m[None, :] * stride_o1)
    tl.store(out_ptrs, acc, mask=(offs_h[:, None] < H) & (offs_m[None, :] < M))


# Main forward: all computation done in Triton
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        device = q_nope.device

        # Shapes
        num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
        num_qo_heads_kpe = q_pe.shape[1]
        assert num_qo_heads == 16, "num_qo_heads must be 16"
        assert head_dim_ckv == 512, "head_dim_ckv must be 512"
        assert num_qo_heads_kpe == 16, "q_pe must have 16 heads"
        assert q_pe.shape[-1] == 64, "head_dim_kpe must be 64"
        num_pages, _, _ = ckv_cache.shape
        assert ckv_cache.shape[1] == 64, "KV page size must be 64"
        assert kpe_cache.shape[1] == 64, "kpe_cache second dim must be 64"
        topk = sparse_indices.shape[-1]
        assert topk == 2048, "topk must be 2048"

        # Prepare flat KV caches as float32 contiguous
        Kc_all = ckv_cache.reshape(-1, 512).to(torch.float32)  # [num_tokens*64, 512]
        Kp_all = kpe_cache.reshape(-1, 64).to(torch.float32)  # [num_tokens*64, 64]
        Kc_all = Kc_all.contiguous()
        Kp_all = Kp_all.contiguous()

        # Output and LSE tensors
        output = torch.zeros(
            (num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
        )
        lse = torch.full((num_tokens, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        for t in range(num_tokens):
            indices = sparse_indices[t]  # [topk]
            valid_mask = indices != -1
            valid_indices = indices[valid_mask]
            M = valid_indices.numel()

            if M == 0:
                # No valid KV for this token: output zeros, lse full -inf
                # Fill lse with -inf to match original behavior
                lse[t].fill_(-float("inf"))
                continue

            # Gather Kc and Kp for this token
            tok_idx = valid_indices.to(torch.long)  # [M]
            Kc_gather = Kc_all[tok_idx]  # [M, 512]
            Kp_gather = Kp_all[tok_idx]  # [M, 64]

            # qn, qp as float32 contiguous
            qn = q_nope[t].to(torch.float32).contiguous()  # [16, 512]
            qp = q_pe[t].to(torch.float32).contiguous()   # [16, 64]

            # Compute logits_qn = qn @ Kc_gather.T → [16, M]
            logits_qn = torch.empty((num_qo_heads, M), dtype=torch.float32, device=device)
            grid_qn = (num_qo_heads // 16, (M + 64 - 1) // 64)
            matmul_qn_kg_kernel[grid_qn](
                qn, Kc_gather, logits_qn,
                H=16, M=M, Kq=512,
                stride_q0=qn.stride(0), stride_q1=qn.stride(1),
                stride_k0=Kc_gather.stride(0), stride_k1=Kc_gather.stride(1),
                stride_o0=logits_qn.stride(0), stride_o1=logits_qn.stride(1),
                BLOCK_H=16, BLOCK_M=64, BLOCK_K=32,
                num_warps=4, num_stages=2
            )

            # Compute logits_qp = qp @ Kp_gather.T → [16, M]
            logits_qp = torch.empty((num_qo_heads, M), dtype=torch.float32, device=device)
            grid_qp = (num_qo_heads // 16, (M + 64 - 1) // 64)
            matmul_qn_kg_kernel[grid_qp](
                qp, Kp_gather, logits_qp,
                H=16, M=M, Kq=64,
                stride_q0=qp.stride(0), stride_q1=qp.stride(1),
                stride_k0=Kp_gather.stride(0), stride_k1=Kp_gather.stride(1),
                stride_o0=logits_qp.stride(0), stride_o1=logits_qp.stride(1),
                BLOCK_H=16, BLOCK_M=64, BLOCK_K=32,
                num_warps=4, num_stages=2
            )

            # C = (logits_qn + logits_qp) * sm_scale
            C = torch.empty_like(logits_qn)
            add_scale_kernel[(num_qo_heads // 16, (M + 256 - 1) // 256)](
                logits_qn, logits_qp, C,
                H=16, M=M,
                stride_a0=logits_qn.stride(0), stride_a1=logits_qn.stride(1),
                stride_b0=logits_qp.stride(0), stride_b1=logits_qp.stride(1),
                stride_c0=C.stride(0), stride_c1=C.stride(1),
                scale=float(sm_scale)
            )

            # Softmax per row (stable) → attn [16, M]
            attn = torch.empty((num_qo_heads, M), dtype=torch.float32, device=device)
            softmax_row_kernel[(num_qo_heads,)](
                C, attn,
                H=16, M=M,
                stride_c0=C.stride(0), stride_c1=C.stride(1),
                stride_a0=attn.stride(0), stride_a1=attn.stride(1)
            )

            # Output = attn @ Kc_gather → [16, 512]
            output_t = torch.empty((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
            _matmul_fp32[(num_qo_heads // 16, (head_dim_ckv + 128 - 1) // 128)](
                attn, Kc_gather,
                output_t,
                M=M, N=512, K=512,
                stride_am=attn.stride(0), stride_ak=attn.stride(1),
                stride_bk=Kc_gather.stride(0), stride_bn=Kc_gather.stride(1),
                stride_cm=output_t.stride(0), stride_cn=output_t.stride(1),
                BLOCK_M=128, BLOCK_N=16, BLOCK_K=32,
                num_warps=4, num_stages=2
            )
            output[t] = output_t.to(torch.bfloat16)

            # LSE per head = logsumexp(C) / ln(2)
            lse_row_kernel[(num_qo_heads,)](
                C, lse[t],
                H=16, M=M,
                stride_c0=C.stride(0), stride_c1=C.stride(1)
            )

        return output, lse


def run(*args):
    return ModelNew()(*args)
