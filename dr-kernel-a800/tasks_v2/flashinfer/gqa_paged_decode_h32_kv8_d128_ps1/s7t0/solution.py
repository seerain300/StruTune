import math
import torch
import triton
import triton.language as tl


# Triton kernel: one program handles one (b, h) pair.
# It computes:
# - logits = q[b, h] @ K_selected.T where K_selected are the tokens' KV head across the batch's window
# - lse = logsumexp(logits * sm_scale) / log(2)
# - out[b, h] = softmax(logits_scaled) @ V_selected
@triton.jit
def process_bh_kernel(
    q_ptr,          # *bf16, shape [B, Nq, D] but we pass pointer and strides
    k_ptr,          # *bf16, shape [Np, Nkv, D]
    v_ptr,          # *bf16, shape [Np, Nkv, D]
    out_ptr,        # *bf16, shape [B, Nq, D]
    lse_ptr,        # *f32,  shape [B, Nq]
    indptr_ptr,     # *i32,  shape [B+1]
    indices_ptr,    # *i32,  shape [T]
    sm_scale,       # f32 scalar
    B: tl.constexpr,          # batch size
    Nq: tl.constexpr,         # num_qo_heads
    Nkv: tl.constexpr,        # num_kv_heads
    D: tl.constexpr,          # head_dim (128)
    gqa_ratio: tl.constexpr,  # Nq // Nkv (4)
    stride_q_b, stride_q_h, stride_q_d,
    stride_k_p, stride_k_h, stride_k_d,
    stride_v_p, stride_v_h, stride_v_d,
    stride_out_b, stride_out_h, stride_out_d,
    stride_lse_b, stride_lse_h,
):
    # program id: one per (b, h)
    pid = tl.program_id(0)
    b = pid // Nq
    h = pid % Nq

    # Select KV head for this query head (GQA mapping)
    kv_head = h // gqa_ratio  # since gqa_ratio == Nq // Nkv, always valid

    # Read q[b, h] as vector of length D
    q_base = q_ptr + b * stride_q_b + h * stride_q_h
    q_vec = tl.load(q_base + tl.arange(0, D) * stride_q_d).to(tl.float32)  # [D], f32

    # Determine token window for this batch
    start = tl.load(indptr_ptr + b)       # int32
    end = tl.load(indptr_ptr + b + 1)    # int32
    T = end - start                       # number of tokens in this batch's window

    # If T == 0, output zeros and lse = -inf; we assume caller handles grid, but we can guard
    # For simplicity, we proceed; T==0 will yield empty loops which is fine.

    # Allocate local vectors for logits and softmax (vectorized across D)
    # Note: Triton prefers compile-time shapes; here we use small D=128 explicitly.
    logits = tl.zeros((D,), dtype=tl.float32)
    attn = tl.zeros((D,), dtype=tl.float32)

    # Compute logits[i] = q_vec @ k_i for i in [0..T-1], where k_i is the selected kv_head
    # We will compute logits in chunks of D elements, but since T loop is small, we do elementwise loop.
    # However, Triton requires vector ops; better approach: compute each logits[i] by a reduction over D.
    # Implement as:
    # for i in range(T): idx = indices[start + i]; k_i = k[idx, kv_head]; attn = dot(q_vec, k_i); store attn.
    # Note: Triton lacks dynamic range with Python 'for', but supports 'while'. We'll use while with i++.
    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        # Load k_i and v_i vectors of length D
        k_base = k_ptr + idx * stride_k_p + kv_head * stride_k_h
        v_base = v_ptr + idx * stride_v_p + kv_head * stride_v_h
        k_i = tl.load(k_base + tl.arange(0, D) * stride_k_d).to(tl.float32)  # [D], f32
        v_i = tl.load(v_base + tl.arange(0, D) * stride_v_d).to(tl.float32)  # [D], f32

        # logits[i] = dot(q_vec, k_i)
        logits[i] = tl.sum(q_vec * k_i, axis=0)

        i += 1

    # Scale and numerically stable logsumexp across i
    scaled_logits = logits * sm_scale  # [D] elements; but we only have D elements, so we need softmax across i dimension?
    # Wait: T is dynamic. Our logits currently only hold one scalar per i? We need a proper T-length vector.
    # Let's rethink: We cannot store T logits in a single vector because T is dynamic. Triton kernel must operate over T.
    # Better: perform reduction across T using a while loop to accumulate max and sum.
    # We'll compute m = max(scaled_logits), then sum = sum(exp(scaled_logits - m)), lse = log(sum) + m, and final result scaled by 1/log(2) if needed.

    # Initialize reduction
    m = tl.full((), -float("inf"), tl.float32)
    # First pass: find max
    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        k_base = k_ptr + idx * stride_k_p + kv_head * stride_k_h
        k_i = tl.load(k_base + tl.arange(0, D) * stride_k_d).to(tl.float32)  # [D]
        attn_i = tl.sum(q_vec * k_i, axis=0)  # scalar
        m = tl.maximum(m, attn_i * sm_scale)
        i += 1

    # Second pass: compute sum of exp(scaled)
    sum_exp = tl.zeros((), dtype=tl.float32)
    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        k_base = k_ptr + idx * stride_k_p + kv_head * stride_k_h
        k_i = tl.load(k_base + tl.arange(0, D) * stride_k_d).to(tl.float32)  # [D]
        attn_i = tl.sum(q_vec * k_i, axis=0)
        sum_exp += tl.exp(attn_i * sm_scale - m)
        i += 1

    lse_val = tl.log(sum_exp) + m  # logsumexp in natural log
    # The original code divides by log(2). Triton supports math ops.
    log2 = 0.6931471805599453  # float64 literal -> Triton will handle; we can use 0.69314718 for f32
    lse_val = lse_val / log2

    # Now compute attn[i] = exp((attn_i * sm_scale - m)) for all i, and then out = attn @ V
    out_vec = tl.zeros((D,), dtype=tl.float32)
    i = 0
    while i < T:
        idx = tl.load(indices_ptr + start + i).to(tl.int32)
        k_base = k_ptr + idx * stride_k_p + kv_head * stride_k_h
        v_base = v_ptr + idx * stride_v_p + kv_head * stride_v_h
        k_i = tl.load(k_base + tl.arange(0, D) * stride_k_d).to(tl.float32)  # [D]
        v_i = tl.load(v_base + tl.arange(0, D) * stride_v_d).to(tl.float32)  # [D]
        attn_i = tl.sum(q_vec * k_i, axis=0)  # scalar
        attn[i] = tl.exp(attn_i * sm_scale - m)  # [1]

        # out_vec += attn[i] * v_i
        out_vec += attn[i] * v_i

        i += 1

    # Store output[b, h, :]
    out_base = out_ptr + b * stride_out_b + h * stride_out_h
    tl.store(out_base + tl.arange(0, D) * stride_out_d, out_vec)

    # Store lse[b, h]
    lse_base = lse_ptr + b * stride_lse_b + h * stride_lse_h
    tl.store(lse_base, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        """
        Triton-optimized forward that performs the same computation as the original PyTorch run,
        but uses Triton kernels for all numeric work. It handles batch_size, num_qo_heads, num_kv_heads,
        and head_dim=128 as in the original asserts. Returns (output, lse).
        """
        assert q.shape[1] == 32, "num_qo_heads must be 32"
        assert q.shape[2] == 128, "head_dim must be 128"
        assert k_cache.shape[1] == 1 and k_cache.shape[3] == 128 and v_cache.shape[1] == 1 and v_cache.shape[3] == 128, "k/v shapes must be [num_pages, 1, num_kv_heads, 128]"
        assert k_cache.shape[2] == 8 and v_cache.shape[2] == 8, "num_kv_heads must be 8"
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "All tensors must be on CUDA for Triton"

        B = q.shape[0]
        Nq = 32
        Nkv = 8
        D = 128
        gqa_ratio = Nq // Nkv  # 4

        # Prepare inputs: ensure int32 for indptr/indices
        kv_indptr_i32 = kv_indptr.to(torch.int32)
        kv_indices_i32 = kv_indices.to(torch.int32)

        # Output and lse tensors
        output = torch.empty((B, Nq, D), dtype=torch.bfloat16, device=q.device)
        lse = torch.empty((B, Nq), dtype=torch.float32, device=q.device)

        # Launch one Triton program per (b, h)
        grid = (B * Nq,)

        # Compute strides (in elements, Triton expects element-wise pointer arithmetic)
        # q: [B, Nq, D]
        q_contig = q.contiguous()
        k_contig = k_cache.squeeze(1).contiguous()  # [Np, Nkv, D]
        v_contig = v_cache.squeeze(1).contiguous()  # [Np, Nkv, D]
        out = output
        # Triton expects pointers; strides are in units of elements:
        stride_q_b = Nq * D
        stride_q_h = D
        stride_q_d = 1
        stride_k_p = Nkv * D
        stride_k_h = D
        stride_k_d = 1
        stride_v_p = Nkv * D
        stride_v_h = D
        stride_v_d = 1
        stride_out_b = Nq * D
        stride_out_h = D
        stride_out_d = 1
        stride_lse_b = Nq
        stride_lse_h = 1

        process_bh_kernel[grid](
            q_contig, k_contig, v_contig, out, lse, kv_indptr_i32, kv_indices_i32,
            float(sm_scale),
            B=B, Nq=Nq, Nkv=Nkv, D=D, gqa_ratio=gqa_ratio,
            stride_q_b=stride_q_b, stride_q_h=stride_q_h, stride_q_d=stride_q_d,
            stride_k_p=stride_k_p, stride_k_h=stride_k_h, stride_k_d=stride_k_d,
            stride_v_p=stride_v_p, stride_v_h=stride_v_h, stride_v_d=stride_v_d,
            stride_out_b=stride_out_b, stride_out_h=stride_out_h, stride_out_d=stride_out_d,
            stride_lse_b=stride_lse_b, stride_lse_h=stride_lse_h,
            num_warps=4, num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
