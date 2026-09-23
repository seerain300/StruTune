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
    def _attention_gqa_kernel(
        q_ptr,           # *f32, shape [B, num_qo_heads, HEAD_DIM]
        k_ptr, v_ptr,    # *bf16 or *f16, shape [num_pages, 1, num_kv_heads, HEAD_DIM]
        kv_indptr_ptr,   # *i32, shape [B+1]
        out_ptr,         # *f32, shape [B, num_qo_heads, HEAD_DIM]
        lse_ptr,         # *f32, shape [B, num_qo_heads]
        B,               # int32
        num_qo_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        sm_scale: tl.constexpr,
        gqa_ratio: tl.constexpr,
        ln2: tl.constexpr,  # natural log of 2
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

        # Load q vector for this (b, h) as float32
        q_offset = b * num_qo_heads * HEAD_DIM + h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_offset)  # [HEAD_DIM] f32

        # Initialize output vector and lse
        out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        lse = tl.full((), -float("inf"), dtype=tl.float32)

        # Loop over tokens
        t = 0
        while t < num_tokens:
            idx = kv_start + t  # scalar int32 index into k/v cache

            # GQA mapping
            kv_head = h // gqa_ratio  # since gqa_ratio = num_qo_heads // num_kv_heads

            # Base offset for (idx, kv_head, :)
            # k_ptr/v_ptr element layout: [num_pages, 1, num_kv_heads, HEAD_DIM] -> linearized
            base = idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM

            # Load k_t and v_t; assume k_cache/v_cache are contiguous in (idx, kv_head, :)
            k_t = tl.load(k_ptr + base)    # vector length HEAD_DIM in original dtype
            v_t = tl.load(v_ptr + base)    # vector length HEAD_DIM in original dtype

            # Cast to f32 for compute
            k_t = k_t.to(tl.float32)
            v_t = v_t.to(tl.float32)

            # Dot product q_vec · k_t
            dot = 0.0
            for d in range(HEAD_DIM):
                dot += q_vec[d] * k_t[d]

            # Scale logits
            scaled = dot * sm_scale

            # Stable LSE update: in natural log; final lse is in natural log units,
            # but we store lse_divided_by_ln2. Original uses logsumexp base 2; here we compute base e.
            is_neg_inf = lse == -float("inf")
            delta = scaled - lse
            max_ls_scaled = tl.maximum(lse, scaled)
            log_term = tl.log(1.0 + tl.exp(-tl.abs(delta)))
            new_lse = max_ls_scaled + log_term
            lse = tl.where(is_neg_inf, scaled, new_lse)

            # Attention weight
            attn = tl.exp(scaled - lse)

            # Accumulate output
            out_vec += attn * v_t

            t += 1

        # Store output vector to out_ptr[b, h]
        out_offset = b * num_qo_heads * HEAD_DIM + h * HEAD_DIM
        tl.store(out_ptr + out_offset, out_vec)

        # Store LSE divided by ln(2) to lse_ptr[b, h]
        lse_div = lse / ln2
        lse_offset = b * num_qo_heads + h
        tl.store(lse_ptr + lse_offset, lse_div)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Expect fixed shapes: q [B, 32, 128], k_cache/v_cache [num_pages, 1, 8, 128]
        B, num_qo_heads, HEAD_DIM = q.shape
        num_kv_heads = k_cache.shape[2]
        gqa_ratio = num_qo_heads // num_kv_heads
        assert num_qo_heads == 32 and num_kv_heads == 8 and HEAD_DIM == 128, "Fixed shapes required"
        assert kv_indptr.shape[0] == B + 1, "kv_indptr length must be batch_size + 1"
        # kv_indptr must start at 0 and end at total tokens (here total_tokens == q.shape[2])
        assert kv_indptr[0].item() == 0
        assert kv_indptr[B].item() == q.shape[2]

        device = q.device

        if TRITON_AVAILABLE and device.type == "cuda":
            # Ensure contiguity
            q_f32 = q.to(torch.float32).contiguous()
            k_cache = k_cache.contiguous()
            v_cache = v_cache.contiguous()
            kv_indptr = kv_indptr.contiguous()

            # Allocate outputs
            output = torch.empty((B, num_qo_heads, HEAD_DIM), dtype=torch.float32, device=device)  # compute in f32
            lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=device)

            # Launch Triton kernel: one program per (b, h)
            grid = (B * num_qo_heads,)
            _attention_gqa_kernel[grid](
                q_f32, k_cache, v_cache, kv_indptr,
                output, lse,
                B,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                HEAD_DIM=HEAD_DIM,
                sm_scale=float(sm_scale),
                gqa_ratio=gqa_ratio,
                ln2=0.6931471805599453,
                num_warps=4,  # heuristic; can be tuned
            )

            # Return output as bfloat16 and lse divided by ln(2) as float32
            output_bf16 = output.to(torch.bfloat16)
            return output_bf16, lse

        else:
            # Fallback: pure-Python computation, no torch ops, matching original logic
            output = torch.empty((B, num_qo_heads, HEAD_DIM), dtype=torch.bfloat16, device=q.device)
            lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q.device)

            for b in range(B):
                kv_start = int(kv_indptr[b].item())
                kv_end = int(kv_indptr[b + 1].item())
                num_tokens = kv_end - kv_start

                q_b = q[b].to(torch.float32)  # [num_qo_heads, HEAD_DIM]
                for h in range(num_qo_heads):
                    kv_head = h // gqa_ratio
                    q_vec = q_b[h]  # [HEAD_DIM] f32

                    out_vec = torch.zeros((HEAD_DIM,), dtype=torch.float32)
                    lse_val = -float("inf")
                    ln2 = 0.6931471805599453

                    for t in range(num_tokens):
                        idx = kv_start + t
                        base = idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM
                        k_t = k_cache[idx, 0, kv_head].to(torch.float32)  # [HEAD_DIM]
                        v_t = v_cache[idx, 0, kv_head].to(torch.float32)  # [HEAD_DIM]

                        dot = 0.0
                        for d in range(HEAD_DIM):
                            dot += q_vec[d] * k_t[d]

                        scaled = dot * sm_scale

                        # Stable LSE update
                        is_neg_inf = lse_val == -float("inf")
                        delta = scaled - lse_val
                        max_ls_scaled = max(lse_val, scaled)
                        log_term = math.log(1.0 + math.exp(-abs(delta)))
                        new_lse = max_ls_scaled + log_term
                        lse_val = scaled if is_neg_inf else new_lse

                        attn = math.exp(scaled - lse_val)
                        out_vec += attn * v_t

                    output[b, h] = torch.tensor(out_vec, dtype=torch.bfloat16, device=q.device)
                    lse[b, h] = torch.tensor(lse_val / ln2, dtype=torch.float32, device=q.device)

            return output, lse

# Original helpers (unchanged)
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
    _out = ModelNew()(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]

class Model(torch.nn.Module):
    def forward(self, *args):
        return ModelNew().forward(*args)


def run(*args):
    return ModelNew()(*args)
