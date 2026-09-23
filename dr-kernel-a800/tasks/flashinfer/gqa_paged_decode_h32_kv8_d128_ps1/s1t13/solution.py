import torch
import math

# Triton availability
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _gqa_attention_kernel(
        q_ptr,           # *f32, shape [B, num_qo_heads, HEAD_DIM]
        k_ptr, v_ptr,    # *bf16 or *f16, shape [num_pages, num_kv_heads, HEAD_DIM]
        kv_indptr_ptr,   # *i32, shape [B+1]
        out_ptr,         # *bf16, shape [B, num_qo_heads, HEAD_DIM]
        lse_ptr,         # *f32, shape [B, num_qo_heads]
        B: tl.constexpr,
        num_qo_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        sm_scale: tl.constexpr,
        gqa_ratio: tl.constexpr,
        ln2: tl.constexpr,  # 1 / ln(2)
    ):
        # One program per (b, h)
        pid = tl.program_id(0)
        b = pid // num_qo_heads
        h = pid % num_qo_heads
        if b >= B or h >= num_qo_heads:
            return

        # Token range for this batch
        kv_start = tl.load(kv_indptr_ptr + b)        # i32
        kv_end = tl.load(kv_indptr_ptr + b + 1)     # i32
        num_tokens = kv_end - kv_start              # i32 scalar

        # Load q vector for this (b, h) as float32: linear offset b*HEAD_DIM*num_qo_heads + h*HEAD_DIM
        q_offset = b * num_qo_heads * HEAD_DIM + h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_offset)  # [HEAD_DIM] f32

        # Initialize output vector and LSE accumulator
        out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        lse = tl.full((), -float("inf"), dtype=tl.float32)
        acc = tl.full((), 0.0, dtype=tl.float32)

        # Loop over tokens
        for t in range(0, num_tokens):
            idx = kv_start + t  # linear index into k/v cache
            kv_head = h // gqa_ratio  # GQA mapping

            # Base offset for k_ptr/v_ptr: layout [num_pages, num_kv_heads, HEAD_DIM]
            # Linearized as [num_pages * num_kv_heads, HEAD_DIM]
            base = idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM

            # Load k_t and v_t vectors of length HEAD_DIM (original cache is bf16/f16; cast to f32 for compute)
            k_t = tl.load(k_ptr + base)           # [HEAD_DIM]
            v_t = tl.load(v_ptr + base)           # [HEAD_DIM]

            # Compute logits = q_vec · k_t
            logits = tl.zeros((), dtype=tl.float32)
            for d in range(HEAD_DIM):
                logits += q_vec[d] * tl.cast(k_t[d], tl.float32)

            # Scale
            scaled = logits * sm_scale  # f32

            # Update LSE stably and acc
            if lse == -float("inf"):
                lse = scaled
                acc = 1.0
            else:
                m = tl.maximum(lse, scaled)
                # new_lse = m + log(exp(lse - m) + exp(scaled - m))
                new_lse = m + tl.log(tl.exp(lse - m) + tl.exp(scaled - m))
                # acc = acc * exp(lse - new_lse) + 1 * exp(scaled - new_lse)
                acc = acc * tl.exp(lse - new_lse) + tl.exp(scaled - new_lse)
                lse = new_lse

            # attention = exp(scaled - lse)
            attn = tl.exp(scaled - lse)
            out_vec += attn * tl.cast(v_t, tl.float32)

        # Store output vector (bfloat16) for this (b, h)
        out_offset = b * num_qo_heads * HEAD_DIM + h * HEAD_DIM
        out_vec_bf16 = tl.cast(out_vec, tl.bfloat16)
        tl.store(out_ptr + out_offset, out_vec_bf16)

        # Store LSE / ln(2) for this (b, h)
        lse_scaled = lse * ln2
        tl.store(lse_ptr + b * num_qo_heads + h, lse_scaled)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Extract shapes
        batch_size, num_qo_heads, head_dim = q.shape
        _, num_pages, num_kv_heads, _ = k_cache.shape

        # Constants
        gqa_ratio = num_qo_heads // num_kv_heads  # 4
        ln2 = 1.0 / math.log(2.0)

        # Ensure contiguity and device; compute in float32
        device = q.device
        q_f32 = q.to(torch.float32).contiguous()
        # k_cache/v_cache may be on CPU in some environments; move to device if Triton is available
        if TRITON_AVAILABLE and device.type == "cuda":
            k_cache = k_cache.to(device=device, dtype=torch.float32).contiguous()
            v_cache = v_cache.to(device=device, dtype=torch.float32).contiguous()
            kv_indptr = kv_indptr.to(device=device)
        else:
            # If not on CUDA, we still set up tensors for kernel launch but keep CPU for safety
            k_cache = k_cache.to(torch.float32).contiguous()
            v_cache = v_cache.to(torch.float32).contiguous()
            kv_indptr = kv_indptr.to(torch.int32)

        # Allocate outputs
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, h)
        grid = (batch_size * num_qo_heads,)
        _gqa_attention_kernel[grid](
            q_f32, k_cache, v_cache, kv_indptr, output, lse,
            B=batch_size,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            HEAD_DIM=head_dim,
            sm_scale=sm_scale,
            gqa_ratio=gqa_ratio,
            ln2=ln2,
        )

        return output, lse

# Optional: keep get_inputs and fused_operator consistent with original API
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
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
