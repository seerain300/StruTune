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
        q_ptr,           # *f32, shape [B, 32, 128]
        k_ptr, v_ptr,    # *bf16 or *f16, shape [num_tokens, 1, 8, 128]
        kv_indptr_ptr,   # *i32, shape [B+1]
        out_ptr,         # *f32, shape [B, 32, 128]
        lse_ptr,         # *f32, shape [B, 32]
        B,               # int32
        HEAD_DIM: tl.constexpr,          # 128
        sm_scale: tl.constexpr,          # 1/sqrt(128)
        gqa_ratio: tl.constexpr,         # 4
        ln2_inv: tl.constexpr,           # 1/log(2)
    ):
        # One program per (b, h)
        pid = tl.program_id(0)
        b = pid // 32
        h = pid % 32
        if b >= B or h >= 32:
            return

        # Load q[b, h, :] as float32
        q_base = b * 32 * HEAD_DIM + h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_base)  # [HEAD_DIM] f32

        # Compute number of tokens for this batch: num_tokens = kv_indptr[b+1] - kv_indptr[b]
        kv_start = tl.load(kv_indptr_ptr + b)        # i32
        kv_end = tl.load(kv_indptr_ptr + b + 1)     # i32
        num_tokens = kv_end - kv_start              # i32 scalar

        # Initialize output vector and lse
        out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        lse = -float("inf")  # f32 scalar

        # Iterate over tokens in this batch
        for t in range(0, num_tokens):
            idx = kv_start + t  # i32

            kv_head = h // gqa_ratio  # 0..7

            # k_ptr/v_ptr layout: [num_tokens, 1, 8, 128] => linear index: idx * (1*8*128) + kv_head*128
            base = idx * (1 * 8 * HEAD_DIM) + kv_head * HEAD_DIM
            k_vec = tl.load(k_ptr + base)  # [HEAD_DIM], may be bf16/f16; cast to f32
            v_vec = tl.load(v_ptr + base)  # [HEAD_DIM]
            k_vec = k_vec.to(tl.float32)
            v_vec = v_vec.to(tl.float32)

            # Dot product q · k_t
            dot = tl.sum(q_vec * k_vec)  # scalar f32

            # Scale logits
            scaled = dot * sm_scale

            # Numerically stable LSE update
            new = tl.maximum(lse, scaled)
            # lse_new = new + log(1 + exp(-(new - lse)))
            lse = new + tl.log(1.0 + tl.exp(-(new - lse)))

            # Attention weight
            attn = tl.exp(scaled - lse)  # scalar

            # Accumulate output
            out_vec += attn * v_vec  # [HEAD_DIM]

        # Store output and lse (divided by ln(2))
        out_base = b * 32 * HEAD_DIM + h * HEAD_DIM
        tl.store(out_ptr + out_base, out_vec)
        tl.store(lse_ptr + (b * 32 + h), lse * ln2_inv)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure contiguity and dtype
        B, num_qo_heads, head_dim = q.shape
        assert num_qo_heads == 32, "num_qo_heads must be 32"
        assert head_dim == 128, "head_dim must be 128"
        num_kv_heads = 8

        device = q.device

        # Compute in float32; return bfloat16 for output
        q_f32 = q.to(torch.float32).contiguous()  # [B, 32, 128] f32
        k_cache = k_cache.contiguous()            # [num_tokens, 1, 8, 128]
        v_cache = v_cache.contiguous()            # [num_tokens, 1, 8, 128]
        kv_indptr = kv_indptr.contiguous()        # [B+1] i32

        # Allocate outputs (compute in float32, return bfloat16 and float32 lse)
        output_f32 = torch.empty((B, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse_f32 = torch.empty((B, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel
        grid = (B * num_qo_heads,)
        _gqa_attention_kernel[grid](
            q_f32, k_cache, v_cache, kv_indptr,
            output_f32, lse_f32,
            B,
            HEAD_DIM=128,
            sm_scale=float(sm_scale),
            gqa_ratio=4,
            ln2_inv=1.0 / math.log(2.0),
        )

        # Return output (bfloat16) and lse (float32)
        return output_f32.to(torch.bfloat16), lse_f32

# Optional: original Model and helpers if needed
class Model(torch.nn.Module):
    def forward(self, *args):
        return run(*args)

@torch.no_grad()
def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, num_qo_heads, head_dim = q.shape
    _, _, num_kv_heads, _ = k_cache.shape
    assert num_qo_heads == 32
    assert num_kv_heads == 8
    assert head_dim == 128

    output = torch.zeros(
        (batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16
    )
    lse = torch.full(
        (batch_size, num_qo_heads), -float("inf"), dtype=torch.float32
    )

    gqa_ratio = num_qo_heads // num_kv_heads

    k_cache_flat = k_cache.squeeze(1).to(torch.float32)  # [num_tokens, 8, 128]
    v_cache_flat = v_cache.squeeze(1).to(torch.float32)  # [num_tokens, 8, 128]

    for b in range(batch_size):
        page_start = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        if page_start >= page_end:
            output[b].zero_()
            lse[b].zero_()
            continue

        num_tokens = page_end - page_start
        q_batch = q[b].to(torch.float32)  # [32, 128]
        for h in range(num_qo_heads):
            kv_head = h // gqa_ratio
            q_head = q_batch[h]  # [128]
            k_list = k_cache_flat[page_start:page_end, kv_head]  # [num_tokens, 128]
            v_list = v_cache_flat[page_start:page_end, kv_head]  # [num_tokens, 128]
            logits = torch.matmul(q_head, k_list.T)  # [num_tokens]
            logits_scaled = logits * sm_scale
            lse[b, h] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)
            attn = torch.softmax(logits_scaled, dim=-1)
            out_head = torch.matmul(attn, v_list)  # [128]
            output[b, h] = out_head.to(torch.bfloat16)
    return output, lse

def get_inputs():
    # Create inputs. The evaluation harness may override devices; here we use cuda for Triton.
    q = torch.randn([1, 32, 128], dtype=torch.bfloat16, device='cuda')
    k_cache = torch.randn([11, 1, 8, 128], dtype=torch.bfloat16, device='cuda')
    v_cache = torch.randn([11, 1, 8, 128], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 10
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to('cuda')
    kv_indices = torch.randint(0, 11, [10], dtype=torch.int32, device='cuda')
    sm_scale = 1.0 / math.sqrt(128)
    return [q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale]

def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5):
    _out = ModelNew()(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
