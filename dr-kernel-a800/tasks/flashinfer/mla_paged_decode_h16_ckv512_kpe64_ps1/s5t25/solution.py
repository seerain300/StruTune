import math
import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, Out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over M and N, loop over K
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def _lse_reduction_kernel(
    logits_ptr, lse_ptr,
    H, L_tokens,
    stride_h, stride_l,
):
    # One program per head; compute logsumexp in natural log
    h = tl.program_id(axis=0)
    max_val = -float('inf')
    for t in range(0, L_tokens):
        ptr = logits_ptr + h * stride_h + t * stride_l
        val = tl.load(ptr)
        if val > max_val:
            max_val = val

    sumexp = 0.0
    for t in range(0, L_tokens):
        ptr = logits_ptr + h * stride_h + t * stride_l
        val = tl.load(ptr)
        sumexp += tl.exp(val - max_val)

    lse = tl.log(sumexp)
    tl.store(lse_ptr + h, lse)


@triton.jit
def _compute_output_kernel(
    logits_ptr, lse_ptr, Kc_ptr, out_ptr,
    H, L_tokens, CK,
    stride_logits_h, stride_logits_l,
    stride_kc_t, stride_kc_k,
    stride_out_h, stride_out_k,
    sm_scale: tl.constexpr,
):
    # One program per head
    h = tl.program_id(axis=0)
    lse_h = tl.load(lse_ptr + h)
    out_acc = tl.zeros((CK,), dtype=tl.float32)

    for t in range(0, L_tokens):
        ptr = logits_ptr + h * stride_logits_h + t * stride_logits_l
        val = tl.load(ptr)  # scaled logits
        softmax = tl.exp(val - lse_h)
        kc_row = tl.load(Kc_ptr + t * stride_kc_t + tl.arange(0, CK) * stride_kc_k)  # [CK]
        out_acc += softmax * kc_row

    out_ptrs = out_ptr + h * stride_out_h + tl.arange(0, CK) * stride_out_k
    tl.store(out_ptrs, out_acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Triton-only forward: compute output tensor only, return single tensor
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda \
               and kv_indptr.is_cuda and kv_indices.is_cuda, "Tensors must be on CUDA device for Triton kernels."

        device = q_nope.device
        dtype_f32 = torch.float32

        # Constants
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]  # 512
        head_dim_kpe = q_pe.shape[2]    # 64
        batch_size = q_nope.shape[0]

        # Selected cache per batch using kv_indptr and kv_indices
        Kc_all = ckv_cache.squeeze(1).to(dtype_f32)  # [num_pages, CK]
        Kp_all = kpe_cache.squeeze(1).to(dtype_f32)  # [num_pages, KP]

        # Output tensor [B, H, CK]
        out = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)

        for b in range(batch_size):
            # Tokens for this batch
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L_tokens = max(page_end - page_beg, 0)

            if L_tokens == 0:
                out[b].zero_()
                continue

            tok_idx = kv_indices[page_beg:page_end].to(torch.int32)  # [L_tokens]
            Kc_selected = Kc_all[tok_idx]  # [L_tokens, CK]
            Kp_selected = Kp_all[tok_idx]  # [L_tokens, KP]

            # A_qn: [H, CK], A_qp: [H, KP]
            A_qn = q_nope[b].to(dtype_f32).reshape(num_qo_heads, head_dim_ckv)  # [H, CK]
            A_qp = q_pe[b].to(dtype_f32).reshape(num_qo_heads, head_dim_kpe)    # [H, KP]

            # B_qn: [CK, L_tokens], B_qp: [KP, L_tokens]
            B_qn = Kc_selected.transpose(0, 1).contiguous()  # [CK, L_tokens]
            B_qp = Kp_selected.transpose(0, 1).contiguous()  # [KP, L_tokens]

            # Intermediate logits buffers
            logits_qn = torch.empty((num_qo_heads, L_tokens), dtype=torch.float32, device=device)
            logits_qp = torch.empty((num_qo_heads, L_tokens), dtype=torch.float32, device=device)

            # Matmul for qn @ Kc.T
            grid_m = (num_qo_heads + 31, ) // 32  # cdiv
            grid_n = (L_tokens + 63, ) // 64
            # Use a dummy third grid dim; Triton will ignore it for matmul
            grid_t = (1,)
            _matmul_kernel[(grid_m, grid_n)](
                A_qn, B_qn, logits_qn,
                num_qo_heads, L_tokens, head_dim_ckv,
                A_qn.stride(0), A_qn.stride(1),
                B_qn.stride(0), B_qn.stride(1),
                logits_qn.stride(0), logits_qn.stride(1),
                BLOCK_M=32, BLOCK_N=64, BLOCK_K=32,
                num_warps=4, num_stages=2,
                H=num_qo_heads, CK=head_dim_ckv, N=L_tokens
            )

            # Matmul for qp @ Kp.T
            grid_m_qp = (num_qo_heads + 31, ) // 32
            grid_n_qp = (L_tokens + 63, ) // 64
            _matmul_kernel[(grid_m_qp, grid_n_qp)](
                A_qp, B_qp, logits_qp,
                num_qo_heads, L_tokens, head_dim_kpe,
                A_qp.stride(0), A_qp.stride(1),
                B_qp.stride(0), B_qp.stride(1),
                logits_qp.stride(0), logits_qp.stride(1),
                BLOCK_M=32, BLOCK_N=64, BLOCK_K=32,
                num_warps=4, num_stages=2,
                H=num_qo_heads, CK=head_dim_kpe, N=L_tokens
            )

            # Scaled logits
            logits_scaled = logits_qn + logits_qp  # [H, L_tokens]
            # For numerical stability, compute lse in Triton; though we don't return it,
            # we keep it in Triton to avoid torch ops and ensure consistency.
            lse = torch.empty((num_qo_heads,), dtype=torch.float32, device=device)
            stride_logits_h = logits_scaled.stride(0)
            stride_logits_l = logits_scaled.stride(1)
            _lse_reduction_kernel[(num_qo_heads,)](
                logits_scaled, lse,
                num_qo_heads, L_tokens,
                stride_logits_h, stride_logits_l,
                H=num_qo_heads, L=L_tokens  # constexpr
            )

            # Final output per head
            out_row = torch.empty((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
            stride_kc_t = Kc_selected.stride(0)
            stride_kc_k = Kc_selected.stride(1)
            stride_out_h = out_row.stride(0)
            stride_out_k = out_row.stride(1)

            _compute_output_kernel[(num_qo_heads,)](
                logits_scaled, lse, Kc_selected,
                out_row,
                num_qo_heads, L_tokens, head_dim_ckv,
                stride_logits_h, stride_logits_l,
                stride_kc_t, stride_kc_k,
                stride_out_h, stride_out_k,
                sm_scale=float(sm_scale),  # constexpr scalar
                H=num_qo_heads, CK=head_dim_ckv, N=L_tokens
            )

            # Assign batch result
            out[b] = out_row

        # Return output as bfloat16 (single tensor), matching original module's output type
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
