import torch
import triton
import triton.language as tl
import math


@triton.jit
def _attention_compute_lse_and_output_kernel(
    qnh_ptr,        # *float32, [512]
    Kc_ptr,         # *float32, [L_tokens, 512], row-major contiguous per row length=512
    Kp_ptr,         # *float32, [L_tokens, 64],  row-major contiguous per row length=64
    qph_ptr,        # *float32, [64]
    out_vec_ptr,    # *float32, [512] output vector for this head
    lse_ptr,        # *float32, [1] scalar for this head
    L_TOKENS: tl.constexpr,   # number of tokens in this batch element (compile-time constant for unrolling)
    SM_SCALE: tl.constexpr,   # scale factor (float)
):
    # Compute logits vector, compute max for numerical stability, and lse
    logits = tl.zeros((L_TOKENS,), dtype=tl.float32)
    # First pass: compute logits and find max
    max_logits = -float("inf")
    for t in tl.static_range(L_TOKENS):
        # Load qnh and qph
        qnh = tl.load(qnh_ptr + tl.arange(0, 512))
        Kc_row = tl.load(Kc_ptr + t * 512 + tl.arange(0, 512))
        qph = tl.load(qph_ptr + tl.arange(0, 64))
        Kp_row = tl.load(Kp_ptr + t * 64 + tl.arange(0, 64))
        # Dot products
        dot1 = tl.sum(qnh * Kc_row, axis=0)
        dot2 = tl.sum(qph * Kp_row, axis=0)
        logits[t] = dot1 + dot2
        max_logits = tl.maximum(max_logits, logits[t])

    # Compute logsumexp_base2 of logits_scaled = logits * SM_SCALE
    sum_exp = 0.0
    for t in tl.static_range(L_TOKENS):
        sum_exp += tl.exp(logits[t] * SM_SCALE - max_logits * SM_SCALE)
    lse_val = max_logits + tl.log(sum_exp) / tl.log(2.0)

    # Store lse
    tl.store(lse_ptr, lse_val)

    # Second pass: compute attention and output
    inv_log2 = 1.0 / tl.log(2.0)
    for t in tl.static_range(L_TOKENS):
        attn_t = tl.exp(logits[t] * SM_SCALE - lse_val * SM_SCALE)
        # Load Kc row
        Kc_row = tl.load(Kc_ptr + t * 512 + tl.arange(0, 512))
        out_vec = attn_t * Kc_row
        # Accumulate into out_vec_ptr
        for d in tl.static_range(512):
            tl.store(out_vec_ptr + d, out_vec[d])


@triton.jit
def _attention_softmax_only_kernel(
    qnh_ptr,        # *float32, [512]
    Kc_ptr,         # *float32, [L_tokens, 512]
    Kp_ptr,         # *float32, [L_tokens, 64]
    qph_ptr,        # *float32, [64]
    attn_ptr,       # *float32, [L_tokens] attention vector for this head
    L_TOKENS: tl.constexpr,
    SM_SCALE: tl.constexpr,
):
    # Compute logits and softmax
    logits = tl.zeros((L_TOKENS,), dtype=tl.float32)
    max_logits = -float("inf")
    for t in tl.static_range(L_TOKENS):
        qnh = tl.load(qnh_ptr + tl.arange(0, 512))
        Kc_row = tl.load(Kc_ptr + t * 512 + tl.arange(0, 512))
        qph = tl.load(qph_ptr + tl.arange(0, 64))
        Kp_row = tl.load(Kp_ptr + t * 64 + tl.arange(0, 64))
        dot1 = tl.sum(qnh * Kc_row, axis=0)
        dot2 = tl.sum(qph * Kp_row, axis=0)
        logits[t] = dot1 + dot2
        max_logits = tl.maximum(max_logits, logits[t])

    sum_exp = 0.0
    for t in tl.static_range(L_TOKENS):
        sum_exp += tl.exp(logits[t] * SM_SCALE - max_logits * SM_SCALE)

    inv_log2 = 1.0 / tl.log(2.0)
    lse_val = max_logits + tl.log(sum_exp) * inv_log2

    for t in tl.static_range(L_TOKENS):
        attn_t = tl.exp(logits[t] * SM_SCALE - lse_val * SM_SCALE)
        tl.store(attn_ptr + t, attn_t)


def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    """
    Triton-orchestrated fused compute. Assumes all inputs are on CUDA and Triton is available.
    q_nope: [B, 16, 512], bfloat16
    q_pe: [B, 16, 64], bfloat16
    ckv_cache: [num_pages, 1, 512], bfloat16
    kpe_cache: [num_pages, 1, 64], bfloat16
    kv_indptr: [batch_size + 1], int32
    kv_indices: [len_indptr[-1] - kv_indptr[0]], int32
    sm_scale: float32
    Returns: output [B, 16, 512], bfloat16, lse [B, 16], float32
    """
    device = q_nope.device
    B = q_nope.shape[0]
    H = q_nope.shape[1]
    D = q_nope.shape[2]

    # Prepare output and lse
    output = torch.zeros((B, H, D), dtype=torch.float32, device=device)
    lse = torch.full((B, H), -float("inf"), dtype=torch.float32, device=device)

    # Process each batch element
    for b in range(B):
        # Number of tokens for this batch element
        start = int(kv_indptr[0].item() if b == 0 else kv_indptr[b].item())
        end = int(kv_indptr[b + 1].item())
        L_tokens = end - start
        if L_tokens <= 0:
            lse[b, :] = -float("inf")
            continue

        # Gather token indices and corresponding rows (cast to float32 for compute)
        tok_idx = kv_indices[start:end]
        Kc_selected = ckv_cache[tok_idx].to(torch.float32)  # [L_tokens, 512]
        Kp_selected = kpe_cache[tok_idx].to(torch.float32)  # [L_tokens, 64]

        # For each head h
        for h in range(H):
            qnh = q_nope[b, h].to(torch.float32)  # [512]
            qph = q_pe[b, h].to(torch.float32)    # [64]
            out_vec = torch.zeros((D,), dtype=torch.float32, device=device)

            # Launch Triton kernel for this (b, h)
            _attention_compute_lse_and_output_kernel[(1,)](
                qnh, Kc_selected, Kp_selected, qph, out_vec, lse[b, h],
                L_TOKENS=L_tokens, SM_SCALE=float(sm_scale)
            )
            output[b, h, :] = out_vec

    # Cast output to bfloat16 as per original signature
    output_bf16 = output.to(torch.bfloat16)
    return output_bf16, lse


def get_inputs():
    # Example inputs, should be on CUDA for Triton
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    num_pages = 989669
    ckv_cache = torch.randn([num_pages, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([num_pages, 1, 64], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 8
    # Construct kv_indptr of length batch_size + 1, and kv_indices length equal to last - first
    kv_indptr = torch.tensor([0, 8], dtype=torch.int32, device='cuda')
    kv_indices = torch.randint(0, num_pages, [8], dtype=torch.int32, device='cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Direct Triton compute: no recursion, no torch softmax/matmul in host
        # Ensure tensors are on CUDA
        if not q_nope.is_cuda:
            q_nope = q_nope.to('cuda')
        if not q_pe.is_cuda:
            q_pe = q_pe.to('cuda')
        if not ckv_cache.is_cuda:
            ckv_cache = ckv_cache.to('cuda')
        if not kpe_cache.is_cuda:
            kpe_cache = kpe_cache.to('cuda')
        if not kv_indptr.is_cuda:
            kv_indptr = kv_indptr.to('cuda')
        if not kv_indices.is_cuda:
            kv_indices = kv_indices.to('cuda')
        return run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)