import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: one program per (b, h). Loops over up to NUM_TOKS with masks.
if TRITON_AVAILABLE:
    @triton.jit
    def _attention_bh_kernel(
        q_ptr,            # *float32, shape [NUM_QO_HEADS, HEAD_DIM]
        k_ptr,            # *float32, shape [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        v_ptr,            # *float32, shape [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        kv_indptr_ptr,    # *int32, shape [BATCH_SIZE+1]
        kv_indices_ptr,   # *int32, shape [NUM_KV_INDICES]
        out_ptr,          # *float32, shape [BATCH_SIZE, NUM_QO_HEADS, HEAD_DIM]
        lse_ptr,          # *float32, shape [BATCH_SIZE, NUM_QO_HEADS]
        sm_scale,         # float32 scalar
        NUM_TOKS: tl.constexpr,  # maximum number of tokens in this batch
        num_tokens,       # int32 scalar: actual number of tokens for this b
        BATCH_SIZE: tl.constexpr,
        NUM_QO_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        b = tl.program_id(0)  # batch id
        h = tl.program_id(1)  # query head id

        # GQA mapping: 32 heads -> 8 kv heads
        kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
        kv_head = h // kv_ratio  # 0..7

        # Load q vector for this head: q[b, h, :]
        q_off = b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_off)  # [HEAD_DIM], float32

        # Compute start index for this batch in kv_indptr
        start = tl.load(kv_indptr_ptr + b)  # int32

        # Pass 1: compute max_s and sum_exp for logsumexp(s) across tokens
        m = -float("inf")  # running max of s
        sumexp = 0.0  # running sum of exp(s - m)
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            # Offsets for k and v rows
            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            # Dot product: logits = q · k_vec
            logits = tl.sum(q_vec * k_vec, axis=0)  # float32
            s = logits * sm_scale  # scaled

            # Update running max and sumexp only if valid
            if mask_i:
                m_new = tl.maximum(m, s)
                # rescale previous sum to new max and add 1 for current
                sumexp = sumexp * tl.exp(m - m_new) + 1.0
                m = m_new

        # lse = log(sumexp) / ln(2) + m
        half_ln2_inv = 1.4426950408889634  # 1 / ln(2)
        lse_val = tl.log(sumexp) * half_ln2_inv + m  # float32
        # Store lse
        tl.store(lse_ptr + b * NUM_QO_HEADS + h, lse_val)

        # Accumulator for output vector
        out_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)

        # Second pass: recompute s, compute attn, and accumulate out
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            # Dot product: logits = q · k_vec
            logits = tl.sum(q_vec * k_vec, axis=0)  # float32
            s = logits * sm_scale

            attn = tl.exp(s - lse_val)  # softmax contribution
            out_vec += attn * v_vec

        # Store output vector for this (b, h)
        out_off = b * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
        tl.store(out_ptr + out_off, out_vec)


def get_inputs():
    q = torch.randn([1, 32, 128], dtype=torch.bfloat16)
    k_cache = torch.randn([11, 1, 8, 128], dtype=torch.bfloat16)
    v_cache = torch.randn([11, 1, 8, 128], dtype=torch.bfloat16)
    _n = 1; _t = 10
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
    kv_indices = torch.randint(0, 11, [10], dtype=torch.int32)
    sm_scale = 1.0 / math.sqrt(128)  # float32 scalar
    return [q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5):
    # TRITON-ONLY implementation: all compute in Triton kernels.
    q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale = tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5
    assert TRITON_AVAILABLE, "Triton is not available"

    # Ensure CUDA tensors and contiguous, cast to float32 for compute
    q = q.contiguous().to(torch.float32).to("cuda")
    k_cache = k_cache.contiguous().to(torch.float32).to("cuda")
    v_cache = v_cache.contiguous().to(torch.float32).to("cuda")
    kv_indptr = kv_indptr.contiguous().to(torch.int32).to("cuda")
    kv_indices = kv_indices.contiguous().to(torch.int32).to("cuda")

    batch_size = q.shape[0]
    num_qo_heads = q.shape[1]
    head_dim = q.shape[2]
    assert num_qo_heads == 32 and head_dim == 128

    # Prepare output and lse (float32), kernel will fill them
    out = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device="cuda")
    lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device="cuda")

    # Pass NUM_TOKS as a meta-parameter. We pick 8192 to cover large workloads safely.
    NUM_TOKS = 8192
    grid = (batch_size, num_qo_heads)

    # Launch kernel per (b, h)
    for b in range(batch_size):
        # Determine actual number of tokens for this batch
        start = int(kv_indptr[b].item())
        end = int(kv_indptr[b + 1].item())
        num_tokens = end - start
        # If no tokens, skip (lse stays at -inf, output zero via store)
        _attention_bh_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, out, lse,
            sm_scale,
            NUM_TOKS=NUM_TOKS,
            num_tokens=num_tokens,
            BATCH_SIZE=batch_size,
            NUM_QO_HEADS=num_qo_heads,
            NUM_KV_HEADS=8,
            HEAD_DIM=head_dim,
            num_warps=4,
        )

    # Cast output to bfloat16 to match original
    out_bf16 = out.to(torch.bfloat16)
    return out_bf16, lse


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Triton implementation: ensure tensors are on CUDA and Triton is available
        if TRITON_AVAILABLE:
            return fused_operator(*args)
        # Fallback to original behavior if Triton is unavailable
        return run(*args)


def run(*args):
    return ModelNew()(*args)
