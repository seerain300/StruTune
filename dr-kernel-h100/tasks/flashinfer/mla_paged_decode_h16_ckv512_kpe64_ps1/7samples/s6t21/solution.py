import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 1: Compute logits for a single (b, h) row:
# logits = (qn @ Kc.T) + (qp @ Kp.T), where qn is 1xHc, qp is 1xHp, Kc is [L,Hc], Kp is [L,Hp]
@triton.jit
def matmul_add_row_kernel(
    qn_ptr, qp_ptr, Kc_ptr, Kp_ptr, out_ptr,
    L_tokens: tl.int32, Hc: tl.int32, Hp: tl.int32, sm_scale: tl.float32,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # qn_ptr: [Hc], qp_ptr: [Hp], Kc_ptr: [L, Hc], Kp_ptr: [L, Hp], out_ptr: [L]
    # We process tokens in chunks of BLOCK_N and reduce over K dims in chunks of BLOCK_K.
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Load qn and qp (1D vectors)
    qn = tl.load(qn_ptr + tl.arange(0, Hc), mask=tl.arange(0, Hc) < Hc, other=0.0)  # [Hc]
    qp = tl.load(qp_ptr + tl.arange(0, Hp), mask=tl.arange(0, Hp) < Hp, other=0.0)  # [Hp]

    # Loop over tokens in chunks
    for n_start in tl.static_range(0, Hc, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)  # output positions in token dim
        mask_n = offs_n < L_tokens
        # Initialize accumulators for this chunk
        acc_chunk_qn = tl.zeros([BLOCK_N], dtype=tl.float32)
        acc_chunk_qp = tl.zeros([BLOCK_N], dtype=tl.float32)

        # Reduce over Hc (K dim for qn @ Kc.T) in blocks
        for k_start in tl.static_range(0, Hc, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
            mask_k = offs_k < Hc
            # Load Kc_chunk: [BLOCK_N, BLOCK_K] using broadcasting
            Kc_chunk = tl.load(
                Kc_ptr + offs_n[:, None] * Hc + offs_k[None, :],
                mask=mask_n[:, None] & mask_k[None, :],
                other=0.0
            )  # [BLOCK_N, BLOCK_K]
            # qn_slice: [BLOCK_K]
            qn_slice = tl.load(qn_ptr + offs_k, mask=mask_k, other=0.0)  # [BLOCK_K]
            acc_chunk_qn += tl.dot(qn_slice, Kc_chunk)  # [BLOCK_N]

        # Reduce over Hp (K dim for qp @ Kp.T) in blocks
        for k_start in tl.static_range(0, Hp, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < Hp
            Kp_chunk = tl.load(
                Kp_ptr + offs_n[:, None] * Hp + offs_k[None, :],
                mask=mask_n[:, None] & mask_k[None, :],
                other=0.0
            )  # [BLOCK_N, BLOCK_K]
            qp_slice = tl.load(qp_ptr + offs_k, mask=mask_k, other=0.0)  # [BLOCK_K]
            acc_chunk_qp += tl.dot(qp_slice, Kp_chunk)  # [BLOCK_N]

        # Accumulate into acc, masked
        acc += tl.where(mask_n, (acc_chunk_qn + acc_chunk_qp) * sm_scale, 0.0)

    # Store final logits for each valid token position
    for i in tl.static_range(0, BLOCK_N):
        idx = n_start + i
        if idx < L_tokens:
            tl.store(out_ptr + idx, acc[i])


# Kernel 2: Compute logsumexp (for lse) of a 1D vector 'in' (length L_tokens) and write to 'out_ptr' (single scalar)
# Uses two passes: pass1 max, pass2 sum(exp(x - max)), then lse = log(sum) / log(2.0)
@triton.jit
def softmax_logsumexp_row_kernel(
    in_ptr, out_ptr, L_tokens: tl.int32, sm_scale: tl.float32,
    BLOCK: tl.constexpr
):
    # Pass 1: compute max
    max_val = -float('inf')
    for start in tl.static_range(0, L_tokens, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < L_tokens
        x = tl.load(in_ptr + offs, mask=mask, other=-float('inf'))
        chunk_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, chunk_max)

    # Pass 2: compute sum of exp(x - max)
    sum_exp = 0.0
    for start in tl.static_range(0, L_tokens, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < L_tokens
        x = tl.load(in_ptr + offs, mask=mask, other=-float('inf'))
        sum_exp += tl.sum(tl.exp(x - max_val), axis=0)

    # lse = log(sum_exp) / log(2.0)
    lse_val = tl.log(sum_exp) / 1.4426950408889634  # 1 / log(2)
    tl.store(out_ptr, lse_val)


# Kernel 3: Compute out_vec = softmax(in_vec) @ Kc, where in_vec is logits (length L_tokens), Kc is [L_tokens, Hc], out_vec is [Hc]
# We process output columns in chunks of BLOCK_N and reduce over tokens in chunks of BLOCK_K.
@triton.jit
def matvec_row_kernel(
    in_ptr, Kc_ptr, out_ptr,
    L_tokens: tl.int32, Hc: tl.int32, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Compute softmax(in_ptr) into attn, then attn @ Kc -> out_ptr
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Pass 1: compute max and sum for numerical stability
    max_val = -float('inf')
    sum_exp = 0.0
    for start in tl.static_range(0, L_tokens, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < L_tokens
        x = tl.load(in_ptr + offs, mask=mask, other=-float('inf'))
        chunk_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, chunk_max)

    for start in tl.static_range(0, L_tokens, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < L_tokens
        x = tl.load(in_ptr + offs, mask=mask, other=-float('inf'))
        sum_exp += tl.sum(tl.exp(x - max_val), axis=0)

    inv_sum = 1.0 / sum_exp

    # Now compute out_vec chunk-wise
    for n_start in tl.static_range(0, Hc, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < Hc
        acc_chunk = tl.zeros([BLOCK_N], dtype=tl.float32)

        # Reduce over tokens in blocks
        for k_start in tl.static_range(0, L_tokens, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < L_tokens
            # Load attn_chunk: [BLOCK_N, BLOCK_K] normalized softmax
            attn_chunk = tl.load(in_ptr + offs_k, mask=mask_k, other=-float('inf'))  # [BLOCK_K]
            attn_chunk = tl.exp(attn_chunk - max_val) * inv_sum  # [BLOCK_K]
            # Load Kc_chunk: [BLOCK_N, BLOCK_K]
            Kc_chunk = tl.load(
                Kc_ptr + offs_n[:, None] * L_tokens + offs_k[None, :],
                mask=mask_n[:, None] & mask_k[None, :],
                other=0.0
            )  # [BLOCK_N, BLOCK_K]
            acc_chunk += tl.sum(attn_chunk[None, :] * Kc_chunk, axis=1)  # [BLOCK_N]

        # Store results for this chunk
        acc += tl.where(mask_n, acc_chunk, 0.0)

    # Store out_vec
    for i in tl.static_range(0, BLOCK_N):
        idx = n_start + i
        if idx < Hc:
            tl.store(out_ptr + idx, acc[i])


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA and dtypes
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All inputs must be on CUDA for Triton."
        device = q_nope.device

        batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
        _, _, head_dim_kpe = q_pe.shape
        num_pages, _, _ = ckv_cache.shape  # expect [num_pages, 1, Hc] but squeeze used in original
        # Squeeze caches to remove the singleton dim
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, Hc]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, Hp]

        # output and lse
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Process each batch element
        for b in range(batch_size):
            # Compute token indices range for this batch
            if kv_indptr.numel() <= 1:
                # Degenerate case, but per original asserts num_indptr >= batch+1; we assume provided correctly.
                continue
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            if start >= end:
                # No tokens for this batch element
                output[b].zero_()
                lse[b].zero_()
                continue

            tok_idx = kv_indices[start:end]  # [L_tokens]
            L_tokens = tok_idx.numel()

            # Gather Kc and Kp for these tokens
            Kc = Kc_all[tok_idx]  # [L_tokens, Hc]
            Kp = Kp_all[tok_idx]  # [L_tokens, Hp]

            # Prepare qn and qp for each head h
            for h in range(num_qo_heads):
                qn = q_nope[b, h].to(torch.float32).contiguous()  # [Hc]
                qp = q_pe[b, h].to(torch.float32).contiguous()   # [Hp]

                # Allocate logits
                logits = torch.empty(L_tokens, dtype=torch.float32, device=device)

                # Kernel 1: compute logits
                BLOCK_N = 128  # token chunk size
                BLOCK_K = 64   # reduction chunk size
                matmul_add_row_kernel[(1,)](
                    qn, qp, Kc, Kp, logits,
                    L_tokens, head_dim_ckv, head_dim_kpe, float(sm_scale),
                    BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                    num_warps=2, num_stages=2
                )

                # Kernel 2: compute lse for this (b, h)
                lse_b_h = lse[b, h]  # 1-element tensor for this (b, h)
                softmax_logsumexp_row_kernel[(1,)](
                    logits, lse_b_h,
                    L_tokens,
                    BLOCK=128,
                    num_warps=1, num_stages=1
                )

                # Kernel 3: compute output[b, h, :]
                out_row = torch.empty(head_dim_ckv, dtype=torch.float32, device=device)
                matvec_row_kernel[(1,)](
                    logits, Kc, out_row,
                    L_tokens, head_dim_ckv,
                    BLOCK_N=128, BLOCK_K=64,
                    num_warps=2, num_stages=2
                )
                output[b, h, :] = out_row

        # Return output in bfloat16, lse in float32
        return output.to(torch.bfloat16), lse


# Helpers from original for testing
def get_inputs():
    # Create random inputs on CUDA
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to(device='cuda')
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32).to(device='cuda')
    sm_scale = 1.0
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    return ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)


def run(*args):
    return ModelNew()(*args)
