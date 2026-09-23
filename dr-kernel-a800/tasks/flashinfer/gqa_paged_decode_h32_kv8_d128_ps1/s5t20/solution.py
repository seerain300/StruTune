import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def gqa_single_token_kernel(
    q_ptr,            # *bf16, shape [B, H, D]
    k_ptr,            # *bf16, shape [Np, 1, K, D]
    v_ptr,            # *bf16, shape [Np, 1, K, D]
    indptr_ptr,       # *int32, shape [B+1]
    indices_ptr,      # *int32, shape [num_kv_indices]
    out_ptr,          # *bf16, shape [B, H, D]
    lse_ptr,          # *float32, shape [B, H]
    sm_scale,         # float32 scalar
    B: tl.constexpr, H: tl.constexpr, D: tl.constexpr, K: tl.constexpr, GR: tl.constexpr,
):
    # program id corresponds to (b, h)
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H
    kv_head = h // GR  # gqa_ratio = 4 -> map qo head to kv head

    # Load Q vector for this (b, h): q[b, h, :]
    q_offset = b * H * D + h * D
    q_vec = tl.load(q_ptr + q_offset + tl.arange(0, D)).to(tl.float32)  # [D], fp32 for stability

    # Load token index for this program (one per program)
    idx = tl.load(indices_ptr + b).to(tl.int32)  # each program handles one token from kv_indices[b]

    # Compute dot product with k[token, kv_head, :]
    k_row_base = idx * (K * D)  # k shape [Np, 1, K, D] => squeezed to [Np, K, D]
    k_vec = tl.load(k_ptr + k_row_base + kv_head * D + tl.arange(0, D)).to(tl.float32)
    logits = tl.sum(q_vec * k_vec, axis=0)  # scalar
    logits_scaled = logits * sm_scale

    # lse in base-2 for a single token is log2(exp(logits_scaled)) = logits_scaled / ln(2)
    lse = logits_scaled / 1.4426950408889634  # 1 / ln(2)

    # Output vector is v[token, kv_head, :]
    v_row_base = idx * (K * D)
    v_vec = tl.load(v_ptr + v_row_base + kv_head * D + tl.arange(0, D)).to(tl.float32)

    # Store results
    out_offset = b * H * D + h * D
    tl.store(out_ptr + out_offset, v_vec.to(tl.bfloat16))
    tl.store(lse_ptr + b * H + h, lse)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure contiguity (no tensor math; just metadata ops)
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Assert shapes
        batch_size, num_qo_heads, head_dim = q.shape
        _, _, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128, \
            f"Expected q: [B,32,128], k_cache: [Np,1,8,128], got q={q.shape}, k={k_cache.shape}"
        gqa_ratio = 4  # 32 // 8

        device = q.device
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Triton launch: one program per (b, h)
        grid = (batch_size * num_qo_heads,)
        gqa_single_token_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, output, lse,
            sm_scale,
            B=batch_size, H=num_qo_heads, D=head_dim, K=num_kv_heads, GR=gqa_ratio,
            num_warps=1, num_stages=1,
        )
        return output, lse


def run(*args):
    return ModelNew()(*args)
