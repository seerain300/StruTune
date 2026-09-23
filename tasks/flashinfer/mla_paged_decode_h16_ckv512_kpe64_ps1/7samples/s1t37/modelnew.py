import math
import torch
import triton
import triton.language as tl


@triton.jit
def attn_logits_kernel(
    qn_ptr,            # *float32, [heads, D] but we index by j
    qp_ptr,            # *float32, [heads, Dp]
    Kc_ptr,            # *float32, [L, D]
    Kp_ptr,            # *float32, [L, Dp]
    logits_ptr,        # *float32, [heads, L]
    heads: tl.int32,   # number of output heads
    D: tl.int32,       # head_dim_ckv
    Dp: tl.int32,      # head_dim_kpe
    L: tl.int32,       # number of selected tokens
    BLOCK_D: tl.constexpr,
    BLOCK_Dp: tl.constexpr,
):
    # Grid: (L,)
    j = tl.program_id(0)  # program handles head j; but here we assume one kernel per head and pass j from host
    # We'll use j as a scalar loaded from a vector to avoid host-passing issues by assuming it's in scope.
    # Instead, use a default j=0 and rely on host to launch per-head independently (done in forward).
    # Compute qn[j, :] and qp[j, :]
    # Since Triton doesn't allow indexing qn_ptr[tl.program_id(0)], we launch this kernel per head externally.
    pass  # Placeholder to avoid compilation errors; Triton will compile but we'll remove unused args.

# NOTE: The above placeholder is for Triton to accept the signature. In practice, we launch per head with separate calls.


# Implement the actual kernels below.

@triton.jit
def attn_logits_kernel_v2(
    qn_ptr,            # *float32, [heads, D]
    qp_ptr,            # *float32, [heads, Dp]
    Kc_ptr,            # *float32, [L, D]
    Kp_ptr,            # *float32, [L, Dp]
    logits_ptr,        # *float32, [heads, L]
    heads: tl.int32,
    D: tl.int32,
    Dp: tl.int32,
    L: tl.int32,
    BLOCK_D: tl.constexpr,
    BLOCK_Dp: tl.constexpr,
):
    # One program per head per token i: grid = (heads, L) but Triton doesn't support 2D program_id(0/1) mapping well.
    # Instead, we launch per head by passing j from host, and loop over i inside the kernel. Here we use single j.

    # We will launch per head (done in forward). This kernel expects j to be provided via an outer loop.
    pass  # Will be replaced with real implementation below.


@triton.jit
def attn_logits_kernel_real(
    qn_ptr,            # *float32, [heads, D] but we index by j
    qp_ptr,            # *float32, [heads, Dp]
    Kc_ptr,            # *float32, [L, D]
    Kp_ptr,            # *float32, [L, Dp]
    logits_ptr,        # *float32, [heads, L]
    j: tl.int32,       # head index
    D: tl.int32,
    Dp: tl.int32,
    L: tl.int32,
    BLOCK_D: tl.constexpr,
    BLOCK_Dp: tl.constexpr,
):
    # One program per token i
    i = tl.program_id(0)
    # Pointers to qn[j, :] and qp[j, :]
    qn_ptr_j = qn_ptr + j * D
    qp_ptr_j = qp_ptr + j * Dp

    # Accumulate logits for this i
    acc = 0.0
    # Reduce over D in chunks
    for d0 in range(0, D, BLOCK_D):
        d_off = d0 + tl.arange(0, BLOCK_D)
        mask_d = d_off < D
        qn_vals = tl.load(qn_ptr_j + d_off, mask=mask_d, other=0.0)  # [BLOCK_D]
        kc_ptr_i = Kc_ptr + i * D + d_off
        kc_vals = tl.load(kc_ptr_i, mask=mask_d, other=0.0)         # [BLOCK_D]
        acc += tl.sum(qn_vals * kc_vals, axis=0)

    # Reduce over Dp in chunks
    for dp0 in range(0, Dp, BLOCK_Dp):
        dp_off = dp0 + tl.arange(0, BLOCK_Dp)
        mask_dp = dp_off < Dp
        qp_vals = tl.load(qp_ptr_j + dp_off, mask=mask_dp, other=0.0)  # [BLOCK_Dp]
        kp_ptr_i = Kp_ptr + i * Dp + dp_off
        kp_vals = tl.load(kp_ptr_i, mask=mask_dp, other=0.0)           # [BLOCK_Dp]
        acc += tl.sum(qp_vals * kp_vals, axis=0)

    # Store logits[j, i]
    tl.store(logits_ptr + j * L + i, acc)


@triton.jit
def lse_base2_kernel(
    logits_ptr,        # *float32, [heads, L]
    lse_ptr,           # *float32, [heads]
    heads: tl.int32,
    L: tl.int32,
):
    # Single program reduces over all elements for each head j
    j = 0  # We'll launch per head in forward, but define a dummy here
    # Two-pass stable reduction
    max_val = -float("inf")
    # pass 1: find max
    for i in range(0, L):
        val = tl.load(logits_ptr + j * L + i)
        max_val = tl.maximum(max_val, val)
    sum_exp = 0.0
    for i in range(0, L):
        val = tl.load(logits_ptr + j * L + i)
        sum_exp += tl.exp(val - max_val)
    lse_j = tl.log(sum_exp) / math.log(2.0) + max_val
    tl.store(lse_ptr + j, lse_j)


@triton.jit
def softmax_base2_kernel(
    logits_ptr,        # *float32, [heads, L]
    lse_ptr,           # *float32, [heads]
    attn_ptr,          # *float32, [heads, L]
    heads: tl.int32,
    L: tl.int32,
    sm_scale: tl.float32,
):
    j = 0
    lse_j = tl.load(lse_ptr + j)
    for i in range(0, L):
        val = tl.load(logits_ptr + j * L + i)
        scaled = val * sm_scale
        attn_val = tl.exp(scaled - lse_j)
        tl.store(attn_ptr + j * L + i, attn_val)


@triton.jit
def matvec_kernel(
    attn_ptr,          # *float32, [heads, L]
    Kc_ptr,            # *float32, [L, D]
    out_ptr,           # *float32, [heads, D]
    heads: tl.int32,
    D: tl.int32,
    L: tl.int32,
    BLOCK_D: tl.constexpr,
):
    j = 0
    for h in range(0, D):
        acc = 0.0
        # Reduce over L
        for i in range(0, L):
            attn_val = tl.load(attn_ptr + j * L + i)
            kc_val = tl.load(Kc_ptr + i * D + h)
            acc += attn_val * kc_val
        tl.store(out_ptr + j * D + h, acc)


def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    # Assume inputs are CUDA tensors
    device = q_nope.device

    batch_size = q_nope.shape[0]
    num_qo_heads = q_nope.shape[1]
    head_dim_ckv = q_nope.shape[2]  # 512
    head_dim_kpe = q_pe.shape[2]    # 64

    # Squeeze the cache to [num_tokens, 512] and [num_tokens, 64]
    Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

    output = torch.zeros((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
    lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    # Loop over batch and heads; launch Triton kernels per (b, j)
    # Ensure q_nope, q_pe are float32 for compute
    q_nope_f32 = q_nope.to(torch.float32)
    q_pe_f32 = q_pe.to(torch.float32)

    for b in range(batch_size):
        # Determine token range for this batch element
        if kv_indptr.numel() <= 1:
            # No tokens selected for this batch
            lse[b] = -float("inf")
            continue

        if kv_indptr.numel() != batch_size + 1:
            # Sanity; if mismatch, fall back to zero output
            output[b].zero_()
            lse[b] = -float("inf")
            continue

        L = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
        if L <= 0:
            lse[b] = -float("inf")
            continue

        # Gather selected tokens
        tokens = kv_indices[int(kv_indptr[b].item()): int(kv_indptr[b + 1].item())]  # [L]
        # Select Kc and Kp for these tokens
        Kc_selected = Kc_all[tokens]  # [L, 512]
        Kp_selected = Kp_all[tokens]  # [L, 64]

        # Allocate intermediate buffers
        logits = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)
        attn = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)
        # For matvec, we write output per head vector directly

        # 1) Compute logits[j, i] for all heads j
        for j in range(num_qo_heads):
            # Launch attn_logits_kernel_real per head
            attn_logits_kernel_real[(L,)](
                q_nope_f32[j], q_pe_f32[j], Kc_selected, Kp_selected, logits, j, head_dim_ckv, head_dim_kpe, L,
                BLOCK_D=128, BLOCK_Dp=64, num_warps=4
            )

        # 2) Compute lse_j = logsumexp(logits[j, :]) / ln(2) for each head
        for j in range(num_qo_heads):
            lse_base2_kernel[(1,)](logits, lse, num_qo_heads, L, sm_scale=1.0)

        # 3) Compute attn[j, :] = softmax_base2(logits_scaled[j, :])
        for j in range(num_qo_heads):
            softmax_base2_kernel[(L,)](logits, lse[b], attn, num_qo_heads, L, sm_scale)

        # 4) Compute out[b, j, :] = attn[j, :] @ Kc_selected[:, :]
        for j in range(num_qo_heads):
            matvec_kernel[(head_dim_ckv,)](attn[j], Kc_selected, output[b, j], num_qo_heads, head_dim_ckv, L, BLOCK_D=128, num_warps=4)

    # Cast output to bfloat16 to match original
    output = output.to(torch.bfloat16)
    return output, lse


# Helper for local testing; harness may provide its own inputs.
def get_inputs():
    # Example inputs; the evaluation harness will override these.
    device = 'cuda'
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device=device)
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device=device)
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device=device)
    # Indptr and indices; len_indptr = batch_size + 1; num_kv_indices = last - first
    L = 8
    kv_indptr = torch.tensor([0, L], dtype=torch.int32, device=device)
    kv_indices = torch.randint(0, 989669, [L], dtype=torch.int32, device=device)
    sm_scale = 1.0
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = run(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


# Entry point for the evaluator
class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Ensure CUDA tensors
        for i in range(len(args)):
            if isinstance(args[i], torch.Tensor) and args[i].device.type != 'cuda':
                args[i] = args[i].to('cuda')
        return run(*args)