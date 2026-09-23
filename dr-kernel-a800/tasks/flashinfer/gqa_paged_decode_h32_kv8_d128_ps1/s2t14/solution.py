import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: one program per (b, h), loops up to NUM_TOKS with masks.
if TRITON_AVAILABLE:
    @triton.jit
    def _attention_bh_kernel(
        q_ptr,            # *float32, [NUM_QO_HEADS, HEAD_DIM]
        k_ptr,            # *float32, [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        v_ptr,            # *float32, [NUM_PAGES, NUM_KV_HEADS, HEAD_DIM]
        kv_indptr_ptr,    # *int32, [BATCH_SIZE+1]
        kv_indices_ptr,   # *int32, [NUM_KV_INDICES]
        out_ptr,          # *float32, [BATCH_SIZE, NUM_QO_HEADS, HEAD_DIM]
        lse_ptr,          # *float32, [BATCH_SIZE, NUM_QO_HEADS]
        B, NQ, NH, KD,    # int32 meta: batch_size, num_qo_heads, num_kv_heads, head_dim
        sm_scale,          # float32 scalar
        NUM_TOKS: tl.constexpr,  # upper bound on tokens (compile-time constant per launch)
    ):
        b = tl.program_id(0)  # batch id
        h = tl.program_id(1)  # query head id

        # Compute GQA mapping: kv_head = h // (32 // 8) = h // 4
        kv_ratio = NQ // NH
        kv_head = h // kv_ratio  # 0..7

        # Determine token range for this batch
        start = tl.load(kv_indptr_ptr + b)         # int32
        end = tl.load(kv_indptr_ptr + b + 1)       # int32
        num_tokens = end - start                    # int32

        # Load q[h, :] vector
        q_off = h * KD  # q is laid out as [NQ, KD]
        q_vec = tl.load(q_ptr + q_off)  # [KD], float32

        # Pass 1: compute max_s and sum_exp across tokens
        m = -float("inf")
        sumexp = 0.0
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            # Offsets for k and v rows (k/v are laid out as [NUM_PAGES, NH, KD])
            k_off = idx * (NH * KD) + kv_head * KD
            v_off = idx * (NH * KD) + kv_head * KD

            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [KD], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [KD], float32

            # Dot product: logits, then scale
            logits = tl.sum(q_vec * k_vec, axis=0)  # float32
            s = logits * sm_scale  # float32

            # Accumulate max and sumexp under mask
            if mask_i:
                m_new = tl.maximum(m, s)
                sumexp = sumexp * tl.exp(m - m_new) + 1.0
                m = m_new

        # lse = log(sumexp) / ln(2) + m
        half_ln2_inv = 1.0 / math.log(2.0)  # float32
        lse_val = tl.log(sumexp) * half_ln2_inv + m  # float32
        # Store lse
        tl.store(lse_ptr + b * NQ + h, lse_val)

        # Accumulator for output vector
        out_vec = tl.zeros([KD], dtype=tl.float32)

        # Second pass: recompute s, compute attn, accumulate out
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            k_off = idx * (NH * KD) + kv_head * KD
            v_off = idx * (NH * KD) + kv_head * KD

            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [KD], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [KD], float32

            logits = tl.sum(q_vec * k_vec, axis=0)  # float32
            s = logits * sm_scale

            attn = tl.exp(s - lse_val)  # float32 scalar
            out_vec += attn * v_vec

        # Store output vector for this (b, h)
        out_off = b * (NQ * KD) + h * KD
        tl.store(out_ptr + out_off, out_vec)


def get_inputs():
    q = torch.randn([1, 32, 128], dtype=torch.bfloat16, device="cuda")
    k_cache = torch.randn([11, 1, 8, 128], dtype=torch.bfloat16, device="cuda")
    v_cache = torch.randn([11, 1, 8, 128], dtype=torch.bfloat16, device="cuda")
    _n = 1; _t = 10
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32, device="cuda")
    kv_indices = torch.randint(0, 11, [10], dtype=torch.int32, device="cuda")
    sm_scale = 1.0 / math.sqrt(128)  # float32 scalar
    return [q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5):
    # Triton-optimized run: all math inside Triton kernels
    q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale = tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5

    assert TRITON_AVAILABLE, "Triton is not available"

    # Ensure inputs are contiguous and float32 for compute
    q = q.contiguous().to(torch.float32)
    k_cache = k_cache.contiguous().to(torch.float32)
    v_cache = v_cache.contiguous().to(torch.float32)
    kv_indptr = kv_indptr.contiguous()
    kv_indices = kv_indices.contiguous()

    batch_size = q.shape[0]
    num_qo_heads = q.shape[1]
    head_dim = q.shape[2]
    num_pages = k_cache.shape[0]  # typically >= number of tokens per batch
    num_kv_heads = k_cache.shape[2]
    assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128, "Shape mismatch: expecting (32,8,128)"

    # Allocate outputs (float32 for compute)
    output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
    lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=q.device)

    # Choose upper bound for tokens; mask handles runtime num_tokens
    NUM_TOKS = 16384  # conservative upper bound; should cover all provided axes

    # Launch Triton kernel: one program per (b, h)
    grid = (batch_size, num_qo_heads)
    _attention_bh_kernel[grid](
        q, k_cache, v_cache, kv_indptr, kv_indices, output, lse,
        batch_size, num_qo_heads, num_kv_heads, head_dim,
        sm_scale,
        NUM_TOKS=NUM_TOKS,
        num_warps=4,
    )

    # Cast output to bfloat16 to match original
    output_bf16 = output.to(torch.bfloat16)

    return output_bf16, lse


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        return fused_operator(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale)


def run(*args):
    return ModelNew()(*args)
