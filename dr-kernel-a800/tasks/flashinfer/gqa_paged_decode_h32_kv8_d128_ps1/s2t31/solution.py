import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: one program per (batch b, query head h).
# It loops over up to NUM_TOKS token indices with masks, accumulating:
# - lse = logsumexp(s) / ln(2), where s = (q[h] · k_i) * sm_scale
# - output vector out = sum_i exp(s - lse) * v_i
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
        B: tl.constexpr,                # batch size
        NUM_QO_HEADS: tl.constexpr,     # 32
        NUM_KV_HEADS: tl.constexpr,     # 8
        HEAD_DIM: tl.constexpr,         # 128
        NUM_TOKS: tl.constexpr,         # upper bound, e.g., 128
    ):
        b = tl.program_id(0)  # batch index
        h = tl.program_id(1)  # query head index

        # Load q[h] vector (float32), shape [HEAD_DIM]
        q_base = h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_base + tl.arange(0, HEAD_DIM))  # [HEAD_DIM]

        # Determine num_tokens_actual for this batch
        start = tl.load(kv_indptr_ptr + b)                   # int32
        end = tl.load(kv_indptr_ptr + b + 1)                # int32
        num_tokens_actual = end - start                     # int32

        # Pass 1: compute logsumexp of s = (q·k_i) * sm_scale across tokens
        max_s = -1.0e30  # float32
        sum_exp = 0.0    # float32

        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS  # 4
            kv_head = h // kv_ratio                 # 0..7

            # Offsets for k and v rows
            k_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM  # [int]
            v_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM  # [int]

            # Load k_vec and v_vec (masked)
            k_vec = tl.load(k_ptr + k_off + tl.arange(0, HEAD_DIM), mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            # v_vec is not needed in this pass; only s depends on k_vec
            # Compute dot product: scalar logits
            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar float32
            s = logits * sm_scale

            # Update max and sum-exp only if valid
            if mask_i:
                max_s = tl.maximum(max_s, s)
                sum_exp = sum_exp * tl.exp(max_s - s) + 1.0

        # lse = log(max_s) + log(sum_exp) * (1/ln(2))
        half_ln2_inv = 1.4426950408889634  # 1 / ln(2)
        lse_val = tl.log(max_s) + tl.log(sum_exp) * half_ln2_inv

        # Write lse for this (b, h)
        tl.store(lse_ptr + b * NUM_QO_HEADS + h, lse_val)

        # Pass 2: recompute s, compute attn, and accumulate output
        out_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for i in range(NUM_TOKS):
            mask_i = i < num_tokens_actual
            idx = tl.load(kv_indices_ptr + start + i, mask=mask_i, other=0)  # int32

            kv_ratio = NUM_QO_HEADS // NUM_KV_HEADS
            kv_head = h // kv_ratio

            k_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM
            v_off = idx * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM

            k_vec = tl.load(k_ptr + k_off + tl.arange(0, HEAD_DIM), mask=mask_i, other=0.0)  # [HEAD_DIM], float32
            v_vec = tl.load(v_ptr + v_off + tl.arange(0, HEAD_DIM), mask=mask_i, other=0.0)  # [HEAD_DIM], float32

            logits = tl.sum(q_vec * k_vec, axis=0)  # scalar float32
            s = logits * sm_scale
            attn = tl.exp(s - lse_val)  # scalar float32
            out_vec += attn * tl.sum(v_vec, axis=0)  # scalar float32 accumulate

        # Store output vector for this (b, h)
        out_base = b * NUM_QO_HEADS * HEAD_DIM + h * HEAD_DIM
        tl.store(out_ptr + out_base + tl.arange(0, HEAD_DIM), out_vec, mask=True)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; all compute is Triton

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are CUDA tensors and contiguous; compute in float32
        assert TRITON_AVAILABLE, "Triton is not available"
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "Inputs must be CUDA tensors"
        device = q.device
        B = q.shape[0]
        assert q.shape[1] == 32 and q.shape[2] == 128, "q must be [B, 32, 128]"
        # k_cache, v_cache: [num_pages, 1, 8, 128]
        num_pages = k_cache.shape[0]
        K_NUM_HEADS = k_cache.shape[2]
        D = k_cache.shape[3]
        assert K_NUM_HEADS == 8 and D == 128, "k_cache/v_cache must be [*, 1, 8, 128]"
        # kv_indptr: [B+1], int32
        assert kv_indptr.shape[0] == B + 1 and kv_indptr.dtype == torch.int32, "kv_indptr must be [B+1], int32"
        # kv_indices: [num_kv_indices], int32
        assert kv_indices.dtype == torch.int32, "kv_indices must be int32"

        # Flatten k_cache/v_cache for simple indexing (still contiguous per (idx, kv_head))
        k_cache_f = k_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 8, 128]
        v_cache_f = v_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 8, 128]
        kv_indices_f = kv_indices.contiguous()  # int32

        # Ensure q is float32 and contiguous
        q_f = q.contiguous().to(torch.float32)  # [B, 32, 128]

        # Allocate output and lse
        out_f = torch.empty((B, 32, 128), dtype=torch.float32, device=device)
        lse_f = torch.empty((B, 32), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, 32)
        _attention_bh_kernel[grid](
            q_f, k_cache_f, v_cache_f,
            kv_indptr, kv_indices_f,
            out_f, lse_f,
            sm_scale,
            B=B, NUM_QO_HEADS=32, NUM_KV_HEADS=8, HEAD_DIM=128, NUM_TOKS=128,
            num_warps=4, num_stages=2,
        )

        # Return output in bfloat16 and lse in float32, matching original signatures
        out_bf16 = out_f.to(torch.bfloat16)
        return out_bf16, lse_f


# Example get_inputs from original for testing (optional):
def get_inputs():
    q = torch.randn([1, 32, 128], dtype=torch.bfloat16, device='cuda')
    k_cache = torch.randn([11, 1, 8, 128], dtype=torch.bfloat16, device='cuda')
    v_cache = torch.randn([11, 1, 8, 128], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 10
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to('cuda')
    kv_indices = torch.randint(0, 11, [10], dtype=torch.int32).to('cuda')
    sm_scale = 1.0 / math.sqrt(128)  # float32 scalar
    return [q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5):
    _out = ModelNew()(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


# Optional quick check (run only if you have CUDA and Triton):
# if __name__ == "__main__":
#     model = ModelNew().cuda()
#     q, k, v, kv_indptr, kv_indices, sm_scale = get_inputs()
#     out, lse = model(q, k, v, kv_indptr, kv_indices, sm_scale)
#     print(out.shape, lse.shape)


def run(*args):
    return ModelNew()(*args)
