import math
import torch
import triton
import triton.language as tl


@triton.jit
def attn_logits_kernel(
    qn_ptr,      # *float32, [D] (for a single head)
    qp_ptr,      # *float32, [Dp] (for a single head)
    Kc_ptr,      # *float32, [L, D] row-major
    Kp_ptr,      # *float32, [L, Dp] row-major
    logits_ptr,  # *float32, [L]
    L: tl.int32,             # number of tokens
    D: tl.int32,             # head_dim_ckv = 512
    Dp: tl.int32,            # head_dim_kpe = 64
    SM_SCALE: tl.float32,    # scaling factor
    BLOCK_L: tl.constexpr,   # tile over L
    BLOCK_K: tl.constexpr,   # tile over K (D or Dp)
):
    # One program computes the entire logits vector for this head
    offs = tl.arange(0, BLOCK_L)
    i = 0
    acc = 0.0
    while i < L:
        idx = i + offs
        mask_i = idx < L
        # Accumulate contributions from Kc and Kp
        acc_kc = 0.0
        k = 0
        while k < D:
            k_off = k + tl.arange(0, BLOCK_K)
            mask_k = k_off < D
            qn_k = tl.load(qn_ptr + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
            kc_ptr = Kc_ptr + idx * D + k_off
            kc = tl.load(kc_ptr, mask=mask_i[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_L, BLOCK_K]
            # Reduce over BLOCK_K
            acc_kc += tl.sum(kc * qn_k[None, :], axis=1)  # sum over K for each i in idx
            k += BLOCK_K

        acc_kp = 0.0
        k = 0
        while k < Dp:
            k_off = k + tl.arange(0, BLOCK_K)  # use same BLOCK_K as D
            mask_k = k_off < Dp
            qp_k = tl.load(qp_ptr + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
            kp_ptr = Kp_ptr + idx * Dp + k_off
            kp = tl.load(kp_ptr, mask=mask_i[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_L, BLOCK_K]
            acc_kp += tl.sum(kp * qp_k[None, :], axis=1)
            k += BLOCK_K

        acc += tl.sum((acc_kc + acc_kp) * (SM_SCALE / 1.4426950408889634))  # 1/ln(2) = 1.4426950408889634
        i += BLOCK_L
    tl.store(logits_ptr + 0, acc)  # placeholder; see below, we compute per i correctly


# We'll implement proper per-i computation instead of summing all logits; better approach:
# We need per-element logits. The kernel will actually write logits[i] per i.

@triton.jit
def attn_logits_kernel_v2(
    qn_ptr,      # *float32, [D]
    qp_ptr,      # *float32, [Dp]
    Kc_ptr,      # *float32, [L, D]
    Kp_ptr,      # *float32, [L, Dp]
    logits_ptr,  # *float32, [L]
    L: tl.int32,             # number of tokens
    D: tl.int32,             # head_dim_ckv
    Dp: tl.int32,            # head_dim_kpe
    SM_SCALE: tl.float32,    # scaling factor
    BLOCK_L: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # One program per output index i in [0, L)
    i = tl.program_id(0)
    # Reduce over Kc and Kp dims
    acc = 0.0
    # Kc reduction
    k = 0
    while k < D:
        k_off = k + tl.arange(0, BLOCK_K)
        mask_k = k_off < D
        qn_k = tl.load(qn_ptr + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
        kc_ptr = Kc_ptr + i * D + k_off
        kc = tl.load(kc_ptr, mask=mask_k, other=0.0)  # scalar or vector, but we reduce across k via loop
        # The previous attempt loaded a vector; to compute scalar logits[i], we should load kc[i] directly.
        # However Triton doesn't support dynamic scalar loads per i; we use vector reduction and mask i.
        # Fix: we'll compute each i by looping over K in blocks and accumulating.
        # For simplicity and correctness, we implement a scalar-per-i reduction by looping over k with BLOCK_K.
        # This avoids trying to vectorize over i and reduces complexity.
        # We'll implement as a two-phase: first compute acc_kc[i], then acc_kp[i], then store logits[i] = acc_kc[i] + acc_kp[i].
        # To keep it simple, we'll use per-i accumulation with masked vector loads and then sum.

        # Instead, we use a simple per-i scalar accumulation:
        # For each k, load kc[i, k] and sum qn[k]*kc[i,k].
        # Triton does not support arbitrary dynamic indexing like kc_ptr[i] directly; we emulate by loading a single scalar per k.
        # We'll do this by setting BLOCK_L=1 and i = program_id(0), which gives us a kernel that runs L times (grid = L).
        # This avoids the earlier incorrect vector approach.

        # Note: Since Triton requires static shapes, we instead write per i using a grid of (L,) and manually loop over D and Dp.
        # Given D and Dp are small relative to typical len, we implement per i scalar accumulation with masked loads.

        # Initialize per-i accumulators
        acc_kc_i = 0.0
        acc_kp_i = 0.0
        # Loop over k in [0, D) with BLOCK_K
        kk = 0
        while kk < D:
            k_off = kk + tl.arange(0, BLOCK_K)
            mask_k = k_off < D
            qn_k = tl.load(qn_ptr + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
            # For each kk, kc[i, kk] is scalar; we load vector kc and reduce over BLOCK_K, but since mask_k is for kk,
            # we can't directly read a scalar. Therefore, we use the fact that qn_ptr is constant across i and instead
            # compute kc for each kk as a vector load with mask, but since we need scalar kc[i,kk], we fallback to a scalar loop.
            # To keep correctness, we implement scalar accumulation for each kk:
            # We need to load scalar kc[i, kk]. Triton doesn't support dynamic scalar indexing with tl.load. Workaround:
            # Use tl.arange to load a vector and sum. However for scalar per i, this is unnecessary. Instead, we loop with BLOCK_K=1.

            # Simpler approach: set BLOCK_K=1 and iterate exactly D times. Triton allows while loops and scalar indexing.
            # But Triton's tl.load expects pointer arithmetic with arange; we cannot index a scalar. Therefore, we implement
            # per-k accumulation by using a scalar pointer offset and masked scalar loads (not supported). Hence we set BLOCK_K=1 and iterate.

            # We'll set BLOCK_K=1 to force per-element loads. Triton accepts this pattern.
            kk += 1  # not used, kept for clarity

        # The above illustrates the structure. We'll simplify: since Triton does not support dynamic scalar loads of K[i,k]
        # from a row-major [L,K] tensor in a vectorized way for scalar per i, we switch to a CPU/GPU torch implementation
        # for logits and use Triton only for lse and matvec. However, to strictly meet the requirement, we implement Triton
        # kernels for logits, lse, and matvec.

        # Fix: Implement logits using torch, then use Triton for lse and matvec. But the requirement is Triton-only for heavy ops.
        # Therefore, we implement Triton kernels for logits too, using a grid of (L,) and BLOCK_D,Dp loops. Triton will run
        # per i with vectorized loads over D/Dp in blocks. This requires careful masking and arithmetic. For correctness,
        # we'll implement the per i logits kernel properly.

        # Re-define kernel with proper per i accumulation:
        # We'll use a grid of (L,) and iterate over D in blocks of BLOCK_K. Triton allows vectorized loads using arange,
        # and we can compute qn[k] * K[i,k] per block and sum. This requires a per i loop over k with vector loads.
        # Triton supports while loops; we can load kc vectors for each k-block and sum.

        # Final implementation:
        # For each i in grid, loop over k in 0..D-1 with step BLOCK_K:
        #   k_off = k + arange(BLOCK_K); mask_k = k_off < D
        #   qn_k = tl.load(qn_ptr + k_off, mask=mask_k, other=0.0)
        #   kc = tl.load(Kc_ptr + i*D + k_off, mask=mask_k, other=0.0)
        #   acc_kc_i += sum(qn_k * kc)  # sum over vector with masked tail
        #   Similarly for Kp.
        # Then store logits[i] = acc_kc_i + acc_kp_i.

        # Implement above logic correctly here:
        acc_kc_i = 0.0
        kk = 0
        while kk < D:
            k_off = kk + tl.arange(0, BLOCK_K)  # BLOCK_K=128
            mask_k = k_off < D
            qn_k = tl.load(qn_ptr + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
            kc_vec = tl.load(Kc_ptr + i * D + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
            prod = qn_k * kc_vec
            # Reduce over BLOCK_K
            # Use sum of prod (masked elements are 0.0)
            acc_kc_i += tl.sum(prod, axis=0)
            kk += BLOCK_K

        acc_kp_i = 0.0
        kpkk = 0
        while kpkk < Dp:
            k_off = kpkk + tl.arange(0, BLOCK_K)  # use same block; Dp<=64 fits
            mask_k = k_off < Dp
            qp_k = tl.load(qp_ptr + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
            kp_vec = tl.load(Kp_ptr + i * Dp + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
            prod = qp_k * kp_vec
            acc_kp_i += tl.sum(prod, axis=0)
            kpkk += BLOCK_K

        logits_val = acc_kc_i + acc_kp_i
        # Apply scaling
        logits_val = logits_val * SM_SCALE
        tl.store(logits_ptr + i, logits_val)


@triton.jit
def lse_base2_kernel(
    v_ptr,      # *float32, [L]
    lse_ptr,    # *float32, scalar per head
    L: tl.int32,
    BLOCK_L: tl.constexpr,
):
    # Two-pass stable lse: first pass max, second pass sum(exp(v - max)) * (1/ln(2)), then lse = log(sum) + max
    max_v = -float("inf")
    i = 0
    while i < L:
        offs = i + tl.arange(0, BLOCK_L)
        mask = offs < L
        v = tl.load(v_ptr + offs, mask=mask, other=-float("inf"))
        max_v = tl.maximum(max_v, tl.max(v, axis=0))
        i += BLOCK_L

    sum_exp = 0.0
    i = 0
    while i < L:
        offs = i + tl.arange(0, BLOCK_L)
        mask = offs < L
        v = tl.load(v_ptr + offs, mask=mask, other=-float("inf"))
        sum_exp += tl.sum(tl.exp(v - max_v), axis=0)
        i += BLOCK_L

    lse = tl.log(sum_exp) + max_v  # natural log
    inv_ln2 = 1.0 / 1.4426950408889634  # 1 / ln(2)
    lse = lse * inv_ln2
    tl.store(lse_ptr, lse)


@triton.jit
def softmax_base2_kernel(
    v_ptr,      # *float32, [L]
    lse_ptr,    # *float32, scalar lse for this head
    attn_ptr,   # *float32, [L]
    L: tl.int32,
    BLOCK_L: tl.constexpr,
):
    inv_ln2 = 1.0 / 1.4426950408889634  # 1 / ln(2)
    lse = tl.load(lse_ptr)
    i = 0
    while i < L:
        offs = i + tl.arange(0, BLOCK_L)
        mask = offs < L
        v = tl.load(v_ptr + offs, mask=mask, other=0.0)
        attn_vals = tl.exp(v * inv_ln2 - lse)
        tl.store(attn_ptr + offs, attn_vals, mask=mask)
        i += BLOCK_L


@triton.jit
def matvec_kernel(
    attn_ptr,   # *float32, [L]
    K_ptr,      # *float32, [L, D] row-major
    y_ptr,      # *float32, [D]
    D: tl.int32,            # output dimension (512)
    L: tl.int32,            # number of tokens
    BLOCK_D: tl.constexpr,  # tile over D
    BLOCK_L: tl.constexpr,  # tile over L
):
    # One program per output dimension h in [0, D)
    h = tl.program_id(0)
    acc = 0.0
    i = 0
    while i < L:
        offs = i + tl.arange(0, BLOCK_L)
        mask_i = offs < L
        attn_vec = tl.load(attn_ptr + offs, mask=mask_i, other=0.0)  # [BLOCK_L]
        k_off = h + tl.arange(0, BLOCK_D)
        mask_k = k_off < D
        Kh = tl.load(K_ptr + h * L + offs[:, None] * D + k_off[None, :], mask=mask_i[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_L, BLOCK_D]
        # acc += sum(attn_vec * K[:, h] over i)
        prod = Kh * attn_vec[:, None]
        acc += tl.sum(prod, axis=0)  # sum over BLOCK_L
        i += BLOCK_L
    tl.store(y_ptr + h, acc)


def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    # Ensure CUDA tensors
    device = q_nope.device
    # Shapes
    batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    assert num_qo_heads == 16, "num_qo_heads must be 16"
    assert head_dim_ckv == 512, "head_dim_ckv must be 512"
    assert head_dim_kpe == 64, "head_dim_kpe must be 64"
    # Constants
    SM_SCALE = float(sm_scale)

    # Output tensors
    output = torch.empty(
        (batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    # Process each batch b
    for b in range(batch_size):
        # Determine token range
        L = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
        if L <= 0:
            # No valid tokens for this batch; set output zeros and lse to -inf
            output[b].zero_()
            lse[b] = -float("inf")
            continue

        # Gather selected Kc and Kp
        tok_idx = kv_indices[b:b + L].to(torch.long)
        Kc_selected = ckv_cache.squeeze(1).index_select(0, tok_idx).to(torch.float32)  # [L, 512]
        Kp_selected = kpe_cache.squeeze(1).index_select(0, tok_idx).to(torch.float32)  # [L, 64]

        # Per-head processing
        for j in range(num_qo_heads):
            qn = q_nope[b, j].to(torch.float32)  # [512]
            qp = q_pe[b, j].to(torch.float32)   # [64]

            # 1) Compute logits[j, :] = qn @ Kc_selected.T + qp @ Kp_selected.T
            logits = torch.empty(L, dtype=torch.float32, device=device)
            # Launch Triton kernel with grid (L,) to compute per-element logits
            BLOCK_L = 64
            BLOCK_K = 128
            attn_logits_kernel_v2[(L,)](
                qn, qp, Kc_selected, Kp_selected, logits,
                L=L, D=512, Dp=64, SM_SCALE=SM_SCALE,
                BLOCK_L=BLOCK_L, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=2
            )

            # 2) Compute base-2 logsumexp of logits for head j
            lse_j = torch.empty(1, dtype=torch.float32, device=device)
            lse_base2_kernel[(1,)](
                logits, lse_j,
                L=L, BLOCK_L=128,
                num_warps=4, num_stages=2
            )
            lse_j = lse_j[0]  # scalar per head

            # 3) Compute attention weights: attn[j, :] = exp(sm_scale * logits[j, :] / ln(2) - lse_j)
            attn = torch.empty(L, dtype=torch.float32, device=device)
            inv_ln2 = 1.4426950408889634  # ln(2)
            softmax_base2_kernel[(L,)](
                logits, lse_j, attn,
                L=L, BLOCK_L=128,
                num_warps=4, num_stages=2
            )

            # 4) Compute out[b, j, :] = attn[j, :] @ Kc_selected[:, :] (matvec over 512)
            y = torch.empty(512, dtype=torch.float32, device=device)
            matvec_kernel[(512,)](
                attn, Kc_selected, y,
                D=512, L=L, BLOCK_D=128, BLOCK_L=64,
                num_warps=4, num_stages=2
            )

            # Store
            output[b, j, :] = y

            # lse[b, j]
            lse[b, j] = lse_j

    # Cast output to bfloat16 to match original
    output = output.to(torch.bfloat16)

    return output, lse


# Helper (non-recursive) to generate inputs compatible with the harness
def get_inputs():
    # Example inputs; harness will provide its own inputs. This helper is not recursive.
    batch_size = 1
    num_qo_heads = 16
    head_dim_ckv = 512
    head_dim_kpe = 64
    # Randomize lengths according to axes variations; minimal setup for local testing.
    num_tokens = 8  # vary in evaluation harness
    L = num_tokens
    q_nope = torch.randn([batch_size, num_qo_heads, head_dim_ckv], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([batch_size, num_qo_heads, head_dim_kpe], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, head_dim_ckv], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, head_dim_kpe], dtype=torch.bfloat16, device='cuda')
    # Indptr and indices for a single batch
    kv_indptr = torch.tensor([0, L], dtype=torch.int32, device='cuda')
    kv_indices = torch.randint(0, 989669, [L], dtype=torch.int32, device='cuda')
    sm_scale = 1.0
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = run(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


# Entry point for ModelNew; Triton kernels are launched here
class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Ensure CUDA tensors
        for i in range(len(args)):
            if isinstance(args[i], torch.Tensor) and args[i].device.type != 'cuda':
                args[i] = args[i].to('cuda')
        return run(*args)


def run(*args):
    return ModelNew()(*args)
