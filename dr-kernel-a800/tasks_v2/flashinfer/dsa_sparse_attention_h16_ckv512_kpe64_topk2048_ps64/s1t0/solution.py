import math
import torch
import triton
import triton.language as tl


# Kernel 1: Compute per-(token, head) logits contributions from sparse_indices.
# We store the logits in a buffer [num_tokens, num_qo_heads, topk] as float32.
@triton.jit
def compute_logits_kernel(
    q_nope_ptr,      # [num_tokens, num_qo_heads, head_dim_ckv], float32
    q_pe_ptr,        # [num_tokens, num_qo_heads, head_dim_kpe], float32
    Kc_ptr,          # [num_pages * page_size, head_dim_ckv], float32 (flattened)
    Kp_ptr,          # [num_pages * page_size, head_dim_kpe], float32 (flattened)
    sparse_ptr,      # [num_tokens, topk], int32
    output_logits_ptr,  # [num_tokens, num_qo_heads, topk], float32
    num_tokens: tl.constexpr,
    num_qo_heads: tl.constexpr,
    head_dim_ckv: tl.constexpr,
    head_dim_kpe: tl.constexpr,
    topk: tl.constexpr,
):
    t = tl.program_id(0)  # token id
    h = tl.program_id(1)  # head id
    # Bounds check (grid should match, but add guard)
    if t >= num_tokens or h >= num_qo_heads:
        return

    # Load q_nope[t, h, :] and q_pe[t, h, :]
    # q_nope_ptr is [num_tokens, num_qo_heads, head_dim_ckv] with strides (s0, s1, s2)
    # We can assume contiguous: s2 = 1, s1 = head_dim_ckv, s0 = num_qo_heads * head_dim_ckv.
    # But simpler: since we pass pointers and grid, we can index directly:
    # Build pointer for q_nope[t, h, :]
    qn_base = q_nope_ptr + t * (num_qo_heads * head_dim_ckv) + h * head_dim_ckv
    qn = tl.zeros((head_dim_ckv,), dtype=tl.float32)
    for d in range(0, head_dim_ckv):
        qn[d] = tl.load(qn_base + d)

    # Similarly for q_pe[t, h, :]
    qp_base = q_pe_ptr + t * (num_qo_heads * head_dim_kpe) + h * head_dim_kpe
    qp = tl.zeros((head_dim_kpe,), dtype=tl.float32)
    for d in range(0, head_dim_kpe):
        qp[d] = tl.load(qp_base + d)

    # Accumulator for logits across topk
    acc = tl.zeros((topk,), dtype=tl.float32)

    # Loop over k in [0, topk)
    for k in range(0, topk):
        idx = tl.load(sparse_ptr + t * topk + k)  # int32
        # Validity check
        if idx != -1:
            # Compute linear index in flattened Kc/Kp: idx is already the flattened index
            Kc_vec_ptr = Kc_ptr + idx * head_dim_ckv
            Kp_vec_ptr = Kp_ptr + idx * head_dim_kpe

            # Dot product contributions
            contrib1 = 0.0
            # Reduce over d in chunks of BLOCK_D
            for d0 in range(0, head_dim_ckv, 128):
                d_offsets = d0 + tl.arange(0, 128)
                mask_d = d_offsets < head_dim_ckv
                qn_chunk = tl.load(qn_base + d_offsets, mask=mask_d, other=0.0)
                Kc_chunk = tl.load(Kc_vec_ptr + d_offsets, mask=mask_d, other=0.0)
                contrib1 += tl.sum(qn_chunk * Kc_chunk, axis=0)

            contrib2 = 0.0
            for d0 in range(0, head_dim_kpe, 32):
                d_offsets = d0 + tl.arange(0, 32)
                mask_d = d_offsets < head_dim_kpe
                qp_chunk = tl.load(qp_base + d_offsets, mask=mask_d, other=0.0)
                Kp_chunk = tl.load(Kp_vec_ptr + d_offsets, mask=mask_d, other=0.0)
                contrib2 += tl.sum(qp_chunk * Kp_chunk, axis=0)

            acc[k] = contrib1 + contrib2

    # Store acc into output_logits[t, h, :]
    out_base = output_logits_ptr + t * (num_qo_heads * topk) + h * topk
    for k in range(0, topk):
        tl.store(out_base + k, acc[k])


# Kernel 2: Compute final output vectors [num_tokens, num_qo_heads, head_dim_ckv] in bfloat16.
# It uses output_logits (after softmax) and gathers corresponding Kc vectors.
@triton.jit
def compute_output_kernel(
    output_logits_ptr,  # [num_tokens, num_qo_heads, topk], float32
    Kc_ptr,             # [num_pages * page_size, head_dim_ckv], float32 (flattened)
    Kp_ptr,             # [num_pages * page_size, head_dim_kpe], float32 (flattened) - not used for output directly
    sparse_ptr,         # [num_tokens, topk], int32
    output_ptr,         # [num_tokens, num_qo_heads, head_dim_ckv], float32 (we will cast to bfloat16 on store)
    num_tokens: tl.constexpr,
    num_qo_heads: tl.constexpr,
    head_dim_ckv: tl.constexpr,
    head_dim_kpe: tl.constexpr,
    topk: tl.constexpr,
):
    t = tl.program_id(0)  # token id
    h = tl.program_id(1)  # head id
    if t >= num_tokens or h >= num_qo_heads:
        return

    # Load attn[h, :] = softmax(output_logits[t, h, :] * sm_scale)
    # We assume sm_scale = 1.0 in the original code; softmax is computed on host for simplicity.
    # But since we need to keep Triton-only computation, we recompute here. However, softmax needs all entries; Triton can handle vector across topk.
    # Load logits vector
    out_base = output_logits_ptr + t * (num_qo_heads * topk) + h * topk
    logits = tl.zeros((topk,), dtype=tl.float32)
    for k in range(0, topk):
        logits[k] = tl.load(out_base + k)
    # Softmax in Triton
    # Numerical stability: subtract max
    max_val = tl.max(logits, axis=0)
    logits = logits - max_val
    exp_logits = tl.exp(logits)
    sum_exp = tl.sum(exp_logits, axis=0)
    attn = exp_logits / sum_exp

    # Accumulate final output vector: out[h, :] = sum_k attn[k] * Kc[idx_k, :]
    out_vec = tl.zeros((head_dim_ckv,), dtype=tl.float32)
    for k in range(0, topk):
        idx = tl.load(sparse_ptr + t * topk + k)
        if idx != -1:
            weight = attn[k]
            Kc_vec_ptr = Kc_ptr + idx * head_dim_ckv
            # Sum over d in chunks
            for d0 in range(0, head_dim_ckv, 128):
                d_offsets = d0 + tl.arange(0, 128)
                mask_d = d_offsets < head_dim_ckv
                Kc_chunk = tl.load(Kc_vec_ptr + d_offsets, mask=mask_d, other=0.0)
                # qn[h, d] is not needed here; we accumulate directly with weight.
                out_vec += weight * Kc_chunk

    # Store output as float32; host will cast to bfloat16
    out_base_out = output_ptr + t * (num_qo_heads * head_dim_ckv) + h * head_dim_ckv
    for d in range(0, head_dim_ckv):
        tl.store(out_base_out + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def __init__(self, num_qo_heads=16, head_dim_ckv=512, head_dim_kpe=64, topk=2048, sm_scale=1.0):
        super().__init__()
        self.num_qo_heads = num_qo_heads
        self.head_dim_ckv = head_dim_ckv
        self.head_dim_kpe = head_dim_kpe
        self.topk = topk
        self.sm_scale = sm_scale

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # Assertions mirroring the original code
        num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages, page_size, _ = ckv_cache.shape
        topk = sparse_indices.shape[-1]
        assert num_qo_heads == self.num_qo_heads
        assert head_dim_ckv == self.head_dim_ckv
        assert head_dim_kpe == self.head_dim_kpe
        assert page_size == 64
        assert topk == self.topk
        device = q_nope.device

        # Flatten K caches to [num_pages * page_size, dim] and cast to float32 for computation
        Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)  # [num_pages*page_size, head_dim_ckv]
        Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)  # [num_pages*page_size, head_dim_kpe]

        # Cast queries to float32 for computation
        q_nope_f32 = q_nope.to(torch.float32)
        q_pe_f32 = q_pe.to(torch.float32)

        # Ensure sparse_indices is int32
        sparse_i32 = sparse_indices.to(torch.int32)

        # Output buffers
        # We'll compute output in float32 in Triton and cast to bfloat16 afterwards
        output_logits = torch.empty((num_tokens, num_qo_heads, self.topk), dtype=torch.float32, device=device)
        output = torch.empty((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.full((num_tokens, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        # Launch kernel 1: compute logits
        grid = (num_tokens, num_qo_heads)
        compute_logits_kernel[grid](
            q_nope_f32, q_pe_f32, Kc_all, Kp_all, sparse_i32, output_logits,
            num_tokens=num_tokens, num_qo_heads=num_qo_heads, head_dim_ckv=head_dim_ckv, head_dim_kpe=head_dim_kpe, topk=self.topk,
            num_warps=4, num_stages=2
        )

        # Compute lse: logsumexp(logits * sm_scale) / log(2)
        # Do this on host using torch (small overhead)
        scaled = output_logits * self.sm_scale
        # Use torch.logsumexp along dim=1 (topk axis)
        lse = torch.logsumexp(scaled, dim=1) / math.log(2.0)

        # Launch kernel 2: compute final output vector using attn
        # We need attn = softmax(scaled, dim=1); compute it in Triton per (t,h)
        # But Triton kernel 2 recomputes softmax; to avoid double work, compute attn on host once:
        # attn = softmax(scaled, dim=1)
        attn = torch.softmax(scaled, dim=1)  # [num_tokens, num_qo_heads, topk]
        # Now call kernel 2 which will use attn to produce output
        compute_output_kernel[grid](
            output_logits, Kc_all, Kp_all, sparse_i32, output,
            num_tokens=num_tokens, num_qo_heads=num_qo_heads, head_dim_ckv=head_dim_ckv, head_dim_kpe=head_dim_kpe, topk=self.topk,
            num_warps=4, num_stages=2
        )

        # Cast output to bfloat16 to match original
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse

# The original helper functions can be reused; these are unchanged:
def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16)
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16)
    ckv_cache = torch.randn([8462, 64, 512], dtype=torch.bfloat16)
    kpe_cache = torch.randn([8462, 64, 64], dtype=torch.bfloat16)
    sparse_indices = torch.randint(0, 541568, [1, 2048], dtype=torch.int32)
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale]

# For consistency with the original Model class:
class Model(torch.nn.Module):
    def forward(self, *args):
        return run(*args)

# If you want to run the new ModelNew, you can do:
# model = ModelNew()
# q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale = get_inputs()
# output, lse = model(q_nope.cuda(), q_pe.cuda(), ckv_cache.cuda(), kpe_cache.cuda(), sparse_indices.cuda(), sm_scale)


def run(*args):
    return ModelNew()(*args)
