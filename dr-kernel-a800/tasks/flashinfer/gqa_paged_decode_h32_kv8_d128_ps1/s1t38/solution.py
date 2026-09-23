import torch
import math

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _attention_bh_kernel(
        q_ptr,           # *f32, shape [B, num_qo_heads, HEAD_DIM], contiguous
        k_ptr, v_ptr,    # *f16/*bf16, shape [num_pages, 1, num_kv_heads, HEAD_DIM], contiguous
        kv_indptr_ptr,   # *i32, shape [B+1], contiguous
        out_ptr,         # *f32, shape [B, num_qo_heads, HEAD_DIM], contiguous (compute in f32, cast later)
        lse_ptr,         # *f32, shape [B, num_qo_heads], contiguous (compute in f32)
        B: tl.constexpr,
        num_qo_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        sm_scale: tl.constexpr,
        ln2: tl.constexpr,
        gqa_ratio: tl.constexpr,  # 4 in the original model
    ):
        # One program per (b, h)
        pid = tl.program_id(0)
        b = pid // num_qo_heads
        h = pid % num_qo_heads
        if b >= B or h >= num_qo_heads:
            return

        # Determine token range
        kv_start = tl.load(kv_indptr_ptr + b)   # i32
        kv_end = tl.load(kv_indptr_ptr + b + 1) # i32
        num_tokens = kv_end - kv_start          # i32 scalar

        # Load q[b, h] as float32
        q_offset = b * num_qo_heads * HEAD_DIM + h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_offset)  # [HEAD_DIM], f32

        # Initialize output and LSE
        out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        lse = tl.full((), -float("inf"), dtype=tl.float32)

        # Loop over tokens: t in [0, num_tokens)
        for t in range(0, num_tokens):
            idx = kv_start + t  # i32 index into kv_indptr's corresponding cache entries
            kvh = h // gqa_ratio  # i32, GQA mapping

            # Gather k_t and v_t as float32: [HEAD_DIM]
            k_offset = idx * num_kv_heads * HEAD_DIM + kvh * HEAD_DIM
            v_offset = idx * num_kv_heads * HEAD_DIM + kvh * HEAD_DIM

            k_t = tl.load(k_ptr + k_offset).to(tl.float32)  # [HEAD_DIM]
            v_t = tl.load(v_ptr + v_offset).to(tl.float32)  # [HEAD_DIM]

            # Dot product q · k_t
            logits = tl.sum(q_vec * k_t, axis=0)  # scalar f32
            scaled = logits * sm_scale  # scalar f32

            # Numerically stable LSE update
            new_lse = tl.maximum(lse, scaled) + tl.log(1.0 + tl.exp(-tl.abs(lse - scaled)))
            lse = new_lse

            # Attention weight and accumulate
            attn = tl.exp(scaled - lse)  # scalar f32
            out_vec += attn * v_t  # [HEAD_DIM]

        # Store results
        out_offset = b * num_qo_heads * HEAD_DIM + h * HEAD_DIM
        tl.store(out_ptr + out_offset, out_vec)

        lse_scaled = lse / ln2
        tl.store(lse_ptr + b * num_qo_heads + h, lse_scaled)

class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # We assume inputs are CUDA tensors when Triton is used.
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda, "Triton path requires CUDA tensors"
        assert TRITON_AVAILABLE, "Triton not available"

        batch_size, num_qo_heads, head_dim = q.shape
        num_pages, _, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128, "Expected fixed shapes: qo_heads=32, kv_heads=8, head_dim=128"
        gqa_ratio = num_qo_heads // num_kv_heads  # 4

        # Ensure contiguity
        q_f32 = q.to(torch.float32).contiguous()
        k_cache_c = k_cache.contiguous()
        v_cache_c = v_cache.contiguous()
        kv_indptr_c = kv_indptr.contiguous()

        # Allocate outputs (compute in float32, cast later)
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (batch_size * num_qo_heads,)
        _attention_bh_kernel[grid](
            q_f32,
            k_cache_c,
            v_cache_c,
            kv_indptr_c,
            output,
            lse,
            B=batch_size,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            HEAD_DIM=head_dim,
            sm_scale=float(sm_scale),
            ln2=float(math.log(2.0)),
            gqa_ratio=gqa_ratio,
            num_warps=4,
        )

        # Cast output to bfloat16 as per original
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse

# The following helpers are unchanged and can be used to generate inputs:
def get_inputs():
    q = torch.randn([1, 32, 128], dtype=torch.bfloat16, device="cuda")
    k_cache = torch.randn([11, 1, 8, 128], dtype=torch.bfloat16, device="cuda")
    v_cache = torch.randn([11, 1, 8, 128], dtype=torch.bfloat16, device="cuda")
    _n = 1; _t = 10
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to("cuda")
    kv_indices = torch.randint(0, 11, [10], dtype=torch.int32).to("cuda")
    sm_scale = 1.0 / math.sqrt(128)  # float32 scalar
    return [q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale]

def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5):
    return ModelNew()(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5)

# Entry point
class Model(torch.nn.Module):
    def forward(self, *args):
        return fused_operator(*args)


def run(*args):
    return ModelNew()(*args)
