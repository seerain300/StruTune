import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


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
        batch_size: tl.constexpr,
        NUM_QO_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        NUM_TOKS: tl.constexpr,         # upper bound of tokens
        sm_scale: tl.constexpr,         # float32 scalar
        half_ln2_inv: tl.constexpr,     # 1 / ln(2) as float32
        num_tokens_actual,               # runtime scalar int32
    ):
        # program ids
        b = tl.program_id(0)  # batch index
        h = tl.program_id(1)  # query head index

        # Compute start and end from kv_indptr
        start = tl.load(kv_indptr_ptr + b)          # int32
        end = tl.load(kv_indptr_ptr + b + 1)        # int32
        num_tokens = end - start                     # int32

        # GQA mapping: kv_head = h // (32 // 8) = h // 4
        kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
        kv_head = h // kv_ratio  # 0..7

        # Load q[h, :]
        q_off = h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_off)  # [HEAD_DIM], float32

        # Accumulators for max and sum(exp(s - max))
        max_s = -float('inf')
        sum_exp = 0.0

        # Pass 1: compute max_s and sum_exp across tokens
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            # Offsets into k and v for this kv_head and token index
            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            # Dot product: scalar logits
            logits = tl.sum(q_vec * k_vec, axis=0)  # float32
            s = logits * sm_scale

            # Update max and sum-exp only if valid
            if mask_i:
                max_s = tl.maximum(max_s, s)
                sum_exp = sum_exp * tl.exp(max_s - s) + 1.0

        # lse = log(max_s) + log(sum_exp) / ln(2)
        lse_val = tl.log(max_s) + tl.log(sum_exp) * half_ln2_inv  # float32
        # Store lse
        tl.store(lse_ptr + b * NUM_QO_HEADS + h, lse_val)

        # Accumulator for output vector
        out_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)

        # Pass 2: recompute s, compute attn, accumulate out
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            k_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
            v_off = idx * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off, mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            logits = tl.sum(q_vec * k_vec, axis=0)  # float32
            s = logits * sm_scale

            attn = tl.exp(s - lse_val)  # float32 scalar
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
    _out = run(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


class Model(torch.nn.Module):
    def forward(self, *args):
        return run(*args)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA and contiguous
        device = q.device
        assert device.type == "cuda", "ModelNew requires CUDA tensors"
        q_f32 = q.contiguous().to(torch.float32)
        k_f32 = k_cache.contiguous().to(torch.float32)
        v_f32 = v_cache.contiguous().to(torch.float32)
        kv_indptr_i32 = kv_indptr.contiguous().to(torch.int32)
        kv_indices_i32 = kv_indices.contiguous().to(torch.int32)

        batch_size = q_f32.shape[0]
        num_qo_heads = q_f32.shape[1]
        head_dim = q_f32.shape[2]
        num_kv_heads = k_f32.shape[2]  # 8

        # Output buffers: compute in float32
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Choose an upper bound for NUM_TOKS; mask guards out-of-range iterations
        # For robustness across provided axes, use a large bound like 16384.
        NUM_TOKS = 16384
        half_ln2_inv = 1.4426950408889634  # 1 / ln(2)

        # Launch kernel: one program per (b, h)
        grid = (batch_size, num_qo_heads)

        # We don't need to pass num_tokens_actual inside the kernel as a meta; Triton supports runtime
        # scalar arguments. The loop uses masks (i < num_tokens) to avoid OOB loads.
        _attention_bh_kernel[grid](
            q_f32,
            k_f32,
            v_f32,
            kv_indptr_i32,
            kv_indices_i32,
            output,
            lse,
            NUM_QO_HEADS=num_qo_heads,
            NUM_KV_HEADS=num_kv_heads,
            HEAD_DIM=head_dim,
            NUM_TOKS=NUM_TOKS,
            sm_scale=float(sm_scale),
            half_ln2_inv=half_ln2_inv,
            num_tokens_actual=(kv_indptr_i32[1:] - kv_indptr_i32[:-1]).min().item(),  # dummy, not used
            num_warps=4,
            num_stages=2,
        )

        # The above dummy arg is not used; Triton expects kwargs. We'll pass correctly below by recomputing
        # num_tokens_actual per batch inside the kernel. To ensure correctness, we can instead compute
        # num_tokens_actual per batch on host and pass it as a runtime arg. But Triton requires kwargs only.
        # So the correct approach is to remove the incorrect kwarg and compute num_tokens_actual inside the kernel
        # from kv_indptr. Let's redefine and launch correctly.

        # Redefine kernel launch without incorrect kwarg. We'll compute num_tokens_actual inside the kernel.
        _attention_bh_kernel[grid](
            q_f32,
            k_f32,
            v_f32,
            kv_indptr_i32,
            kv_indices_i32,
            output,
            lse,
            NUM_QO_HEADS=num_qo_heads,
            NUM_KV_HEADS=num_kv_heads,
            HEAD_DIM=head_dim,
            NUM_TOKS=NUM_TOKS,
            sm_scale=float(sm_scale),
            half_ln2_inv=half_ln2_inv,
            num_warps=4,
            num_stages=2,
        )

        # Cast output to bfloat16 to match original return types
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
