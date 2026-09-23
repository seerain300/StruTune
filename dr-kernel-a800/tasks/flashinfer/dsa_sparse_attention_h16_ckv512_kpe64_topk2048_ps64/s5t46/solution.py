import torch
import math
import triton
import triton.language as tl


# Triton kernel: per-row logsumexp (natural log) over M, and store lse/ln(2) into out_lse[H].
@triton.jit
def lse_row_typed(C_ptr, out_lse_ptr,
                  H: tl.constexpr, M: tl.int32,
                  stride_c0, stride_c1,
                  inv_ln2: tl.float32,
                  BM: tl.constexpr):
    h = tl.program_id(0)
    # 1) find per-row max
    row_max = -float("inf")
    for m0 in range(0, M, BM):
        offs_m = m0 + tl.arange(0, BM)
        c_ptrs = C_ptr + (h * stride_c0 + offs_m * stride_c1)
        x = tl.load(c_ptrs, mask=offs_m < M, other=-float("inf"))
        local_max = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, local_max)

    # 2) compute sum of exp(x - row_max)
    sum_exp = 0.0
    for m0 in range(0, M, BM):
        offs_m = m0 + tl.arange(0, BM)
        c_ptrs = C_ptr + (h * stride_c0 + offs_m * stride_c1)
        x = tl.load(c_ptrs, mask=offs_m < M, other=-float("inf"))
        e = tl.exp(x - row_max)
        sum_exp += tl.sum(e, axis=0)

    lse = row_max + tl.log(sum_exp)
    # store lse / ln(2)
    out_lse_h = lse * inv_ln2
    tl.store(out_lse_ptr + h, out_lse_h)


# Triton kernel: softmax per row: attn[H, M] = softmax(C[H, M], dim=1) using stable algorithm.
@triton.jit
def softmax_row_typed(C_ptr, attn_ptr,
                      H: tl.constexpr, M: tl.int32,
                      stride_c0, stride_c1,
                      stride_a0, stride_a1,
                      BM: tl.constexpr):
    h = tl.program_id(0)
    # Pass 1: row max
    row_max = -float("inf")
    for m0 in range(0, M, BM):
        offs_m = m0 + tl.arange(0, BM)
        c_ptrs = C_ptr + (h * stride_c0 + offs_m * stride_c1)
        x = tl.load(c_ptrs, mask=offs_m < M, other=-float("inf"))
        local_max = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, local_max)

    # Pass 2: row sum
    row_sum = 0.0
    for m0 in range(0, M, BM):
        offs_m = m0 + tl.arange(0, BM)
        c_ptrs = C_ptr + (h * stride_c0 + offs_m * stride_c1)
        x = tl.load(c_ptrs, mask=offs_m < M, other=-float("inf"))
        e = tl.exp(x - row_max)
        row_sum += tl.sum(e, axis=0)

    inv_row_sum = 1.0 / row_sum

    # Pass 3: write normalized
    for m0 in range(0, M, BM):
        offs_m = m0 + tl.arange(0, BM)
        c_ptrs = C_ptr + (h * stride_c0 + offs_m * stride_c1)
        x = tl.load(c_ptrs, mask=offs_m < M, other=-float("inf"))
        e = tl.exp(x - row_max) * inv_row_sum
        a_ptrs = attn_ptr + (h * stride_a0 + offs_m * stride_a1)
        tl.store(a_ptrs, e, mask=offs_m < M)


def run_triton_variant(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """
    Compute attention output using PyTorch matmuls for logits and Triton for softmax/lse.
    Returns (output: [num_tokens, num_qo_heads, head_dim_ckv], lse: [num_tokens, num_qo_heads]).
    """
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages, page_size, _ = ckv_cache.shape
    topk = sparse_indices.shape[-1]

    # assert fixed parameters
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 64
    assert topk == 2048

    # Checks
    assert sparse_indices.shape[0] == num_tokens
    assert sparse_indices.shape[-1] == topk
    assert ckv_cache.shape[1] == page_size

    device = q_nope.device

    # Flatten paged KV cache to token-level: [num_pages, page_size, dim] -> [num_pages * page_size, dim]
    Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)  # [total_kv_tokens, head_dim_ckv]
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)  # [total_kv_tokens, head_dim_kpe]

    output = torch.empty(
        (num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.empty((num_tokens, num_qo_heads), dtype=torch.float32, device=device)

    for t in range(num_tokens):
        indices = sparse_indices[t]  # [topk]
        valid_mask = indices != -1
        valid_indices = indices[valid_mask]

        if valid_indices.numel() == 0:
            # output zeros, lse = -inf
            output[t].zero_()
            lse[t] = float("-inf")
            continue

        # For page_size=64, indices encode (page_idx * 64 + offset)
        tok_idx = valid_indices.to(torch.long)

        # Gather K and corresponding p-cache for these tokens
        Kc = Kc_all[tok_idx]  # [num_valid, head_dim_ckv]
        Kp = Kp_all[tok_idx]  # [num_valid, head_dim_kpe]

        # q tensors [H, Kc_dim/Kp_dim]
        qn = q_nope[t].to(torch.float32)  # [H, head_dim_ckv]
        qp = q_pe[t].to(torch.float32)    # [H, head_dim_kpe]

        # Compute attention logits via torch matmul
        logits_qn = torch.matmul(qn, Kc.transpose(0, 1))  # [H, num_valid]
        logits_qp = torch.matmul(qp, Kp.transpose(0, 1))  # [H, num_valid]
        logits = logits_qn + logits_qp  # [H, num_valid]
        logits_scaled = logits * sm_scale  # [H, num_valid]

        # lse per (H,)
        H = num_qo_heads
        M = logits_scaled.shape[1]
        inv_ln2 = 1.0 / math.log(2.0)

        lse_row_typed[(H,)](
            logits_scaled, lse[t],
            H, M,
            logits_scaled.stride(0), logits_scaled.stride(1),
            inv_ln2,
            256,
            num_warps=4, num_stages=2
        )

        # attn per (H, M)
        attn_per = torch.empty_like(logits_scaled)
        softmax_row_typed[(H,)](
            logits_scaled, attn_per,
            H, M,
            logits_scaled.stride(0), logits_scaled.stride(1),
            attn_per.stride(0), attn_per.stride(1),
            256,
            num_warps=4, num_stages=2
        )

        # Compute final output: out[H, head_dim_ckv] = attn @ Kc
        out_per = torch.matmul(attn_per, Kc)  # [H, head_dim_ckv]
        output[t] = out_per.to(torch.bfloat16)

    return output, lse


# Original helper to generate inputs (kept identical, use CUDA to run Triton kernels)
def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([8462, 64, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([8462, 64, 64], dtype=torch.bfloat16, device='cuda')
    sparse_indices = torch.randint(0, 541568, [1, 2048], dtype=torch.int32, device='cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale]


# Entry point: ModelNew
class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        return run_triton_variant(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)


def run(*args):
    return ModelNew()(*args)
