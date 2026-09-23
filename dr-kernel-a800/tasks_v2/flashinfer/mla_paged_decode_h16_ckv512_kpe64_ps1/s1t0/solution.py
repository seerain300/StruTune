import torch
import math
import triton
import triton.language as tl


# Kernel 1: For each head i, compute attention weights (attn[i, :]) and logsumexp scaled by ln(2)
@triton.jit
def compute_logits_scaled_and_attn_kernel(
    qn_ptr, qp_ptr, Kc_ptr, Kp_ptr,
    attn_ptr, lse_ptr,
    H: tl.constexpr, Dc: tl.constexpr, Dp: tl.constexpr,
    L_tokens, sm_scale,
    BLOCK_L: tl.constexpr,
):
    i = tl.program_id(0)  # head index
    # Initialize vector for logits_scaled for this head
    # We'll compute it chunk by chunk and then update attn_ptr and lse_ptr
    # Compute lse via a reduction over all tokens
    # We need to iterate over tokens t and compute logits[t] for each chunk
    # We will write attn[i, t] as float32
    # And then compute lse[i] = logsumexp(logits_scaled[i, :]) / ln(2)

    # Precompute ln(2) in float32
    ln2 = 0.6931471805599453  # math.log(2.0) in float64, cast to fp32

    # We'll perform two loops: one to fill attn and then compute lse, and another to store attn (we can store while computing)
    # But Triton doesn't easily allow storing then reducing; better approach: compute attn in chunks and maintain a running max and sum for lse.
    # However, computing lse requires knowing all attn values. Therefore, we first compute and store attn, then compute lse using a second pass.
    # To keep code simple and correct, we'll store attn and compute lse in the same kernel by iterating again.

    # First pass: compute and store attn; also collect max and sum_exp for lse
    m = -float("inf")
    sum_exp = 0.0
    for t_base in range(0, L_tokens, BLOCK_L):
        t = t_base + tl.arange(0, BLOCK_L)
        mask = t < L_tokens

        # Load qn[i, :] and qp[i, :]
        # qn shape: [H, Dc], row i; qp shape: [H, Dp], row i
        # Use strides to load
        # We assume qn_ptr points to [H, Dc] contiguous; stride_qn_d = 1; stride_qn_h = Dc
        # But we pass qn_ptr as flattened [H*Dc], so we need to compute address for row i
        # Let's pass qn as [H, Dc] pointer: not possible in Triton; we'll pass qn_ptr as [H*Dc] contiguous with strides.
        # Simpler: pass qn_ptr as [H, Dc] 2D pointer. But Triton expects 1D. We need to change how we pass qn,qp.
        # To keep it simple, we'll assume we pass qn,qp as [H, Dc/Dp] to host. Triton expects 1D, so we need to create 1D buffers.
        # Therefore, we'll pass qn[i, :] flattened as qn_ptr[i*Dc : (i+1)*Dc] and same for qp.

        # We'll implement by loading qn[i, :] and qp[i, :] as 1D vectors using tl.load with base pointer and arange.
        # First, compute base pointers for qn[i, :], qp[i, :]
        # However, Triton doesn't support passing [H, Dc] pointers directly; we need to flatten and use tl.load with base + arange.
        # Since Triton sees pointers, we'll pass qn_ptr as 1D of size H*Dc, and we'll pass i as program_id and compute base as i*Dc.
        # But program_id is only for heads. We need another way. Instead, we will pass qn_ptr as 2D, but Triton kernel doesn't support 2D.
        # Resolution: we'll pass qn_ptr, qp_ptr as 1D arrays of length H*Dc/H*Dp, and index accordingly.

        # To do that, we must precompute base offsets. Let's redefine kernel signature to accept qn[i, :] as a pointer.
        # Triton supports tl.load with pointer + arange, so we can load qn[i, :] as tl.load(qn_ptr + i*stride, ...).
        # But Triton requires tl.load to have a pointer. The usual pattern: pointer is an argument; then load using pointer + offsets.
        # Here, we'll pass qn_ptr, qp_ptr as 1D, and compute base = i * Dc for qn[i, :], base = i * Dp for qp[i, :].
        # Then, load qn_vec = tl.load(qn_ptr + i * Dc + k*stride), but since we pass 1D qn_ptr of size H*Dc, we can load qn[i, :] with tl.load(qn_ptr + i * Dc + tl.arange(0, Dc), mask=..., other=0.0).
        # However, we don't know Dc at compile time here; we need to pass Dc as tl.constexpr or loop over Dc.

        # Implementing this properly requires redefining how we pass qn, qp. For simplicity, we'll rework the kernel to use 2D pointers.
        # But Triton doesn't support 2D pointer arguments in a straightforward way for this context. Therefore, we will pass qn/qp as 1D and compute via strides.

        # Let's define qn_ptr and qp_ptr as 1D arrays of size H*Dc and H*Dp, respectively. We'll compute base offsets using tl.load.
        # But Triton requires pointer to be a Triton pointer tensor. We can pass flattened tensors and compute base offsets in host by slicing.
        # Resolution: In Python, we allocate qn_flat and qp_flat before kernel launch and pass them. Similarly for Kc_ptr, Kp_ptr.

        # Given that, we redefine kernel arguments to receive 1D qn_vec_ptr, qp_vec_ptr, Kc_ptr, Kp_ptr, and iterate with tl.load.

        # For now, we keep the approach: assume qn_ptr, qp_ptr, Kc_ptr, Kp_ptr are 1D and we can load with base + tl.arange.
        # We'll fix this in the next implementation by passing flattened 1D pointers.

        # Note: The above comment shows a need for careful pointer handling. We'll implement a corrected version below by assuming flattened 1D pointers.

    # Since the above approach is getting tricky, let's implement a simpler, correct version: we'll compute attn by loading chunks from Kc_ptr/Kp_ptr
    # and qn[i, :], qp[i, :] vectors, then store attn and compute lse in a second loop. To keep code size reasonable, we'll implement it explicitly.

    # Correct implementation plan:
    # 1) Compute attn[i, :] in chunks of BLOCK_L:
    #    - For each chunk, initialize acc[BLOCK_L] = 0.0
    #    - Loop over k in 0..Dc-1: load qn[i, k] scalar, then load Kc[t, k] chunk vector, accumulate acc += qn_k * Kc_chunk
    #    - Repeat for k in 0..Dp-1: load qp[i, k] scalar, then load Kp[t, k] chunk vector, accumulate acc += qp_k * Kp_chunk
    #    - Scale acc by sm_scale
    #    - Store acc to attn_ptr[i * L_tokens + t]
    #    - Update m and sum_exp for lse using only valid t
    # 2) After first pass, compute lse[i] = m + log(sum_exp) / ln(2) and store to lse_ptr[i].
    #    Note: We only store lse for heads that have L_tokens > 0. For L_tokens == 0, we skip and host will zero output.

    # Implementing 1):
    for t_base in range(0, L_tokens, BLOCK_L):
        t = t_base + tl.arange(0, BLOCK_L)
        mask_t = t < L_tokens

        # Initialize acc vector for this chunk
        acc = tl.zeros([BLOCK_L], dtype=tl.float32)

        # Accumulate from Kc
        # Loop over k in 0..Dc-1: Triton requires compile-time loop bounds; we pass Dc as tl.constexpr
        for k in range(0, Dc):
            # Load qn[i, k]
            # qn_ptr is 1D with length H*Dc
            qn_k = tl.load(qn_ptr + i * Dc + k, mask=True, other=0.0)
            # Load Kc[t, k] chunk: Kc_ptr length = num_tokens * Dc; Kc[t, :] base at t*Dc + k*1, but we pass it as 1D flattened: index = t*Dc + k
            # We'll pass Kc_ptr as flattened [L_tokens, Dc] -> [L_tokens*Dc], but to keep it simple, we'll assume Kc_ptr is 2D and pass via 1D:
            # Since Triton doesn't support 2D pointer args, we'll restructure and pass Kc_ptr as 1D and compute address as t*Dc + k.
            # However, in Triton kernel we don't have access to the external shapes beyond pointers. Therefore, we'll pass qn_ptr, qp_ptr, Kc_ptr, Kp_ptr as 1D arrays and compute base offsets in host before launch.
            # For clarity, we'll implement the chunk loading by assuming qn_ptr, qp_ptr, Kc_ptr, Kp_ptr are 1D arrays corresponding to flattened rows/cols.
            # Load Kc[t, k] chunk vector: index = t * Dc + k
            Kc_chunk = tl.load(Kc_ptr + t * Dc + k, mask=mask_t, other=0.0)
            acc += qn_k * Kc_chunk

        # Accumulate from Kp
        for k in range(0, Dp):
            qp_k = tl.load(qp_ptr + i * Dp + k, mask=True, other=0.0)
            Kp_chunk = tl.load(Kp_ptr + t * Dp + k, mask=mask_t, other=0.0)
            acc += qp_k * Kp_chunk

        # Scale by sm_scale
        acc = acc * sm_scale

        # Store attn[i, t] = acc
        # attn_ptr is 1D with length H * L_tokens
        attn_offset = i * L_tokens + t
        tl.store(attn_ptr + attn_offset, acc, mask=mask_t)

        # Update m and sum_exp for lse (only valid t)
        # m = max(m, max(acc over valid t))
        # sum_exp += sum(exp(acc - m) over valid t)
        # To do that, we need to know which t are valid. Triton doesn't allow dynamic masks on reductions easily here, so we'll do a second pass.

    # 2) Second pass to compute lse: we need to compute max and sum of exp(acc) over all t; since we stored acc (i.e., attn), we can read them back.
    # But to avoid extra memory, we can maintain running m and sum_exp in registers. However, Triton does not allow global state per program across loops here cleanly.
    # So we recompute max and sum using the stored attn. We'll read attn back. But since we already computed and stored, we can compute lse using a second loop over chunks.

    # However, storing acc and reading it back is not ideal. Therefore, we'll compute lse using the stored attn in a second kernel. But since we need to keep everything in one kernel, we'll restructure.

    # Resolution: We'll compute attn and, for each t in chunks, we'll compute chunk_max = max(acc over chunk) and chunk_sum_exp = sum(exp(acc - chunk_max)), updating global m and sum_exp.
    # Initialize global m = -inf, sum_exp = 0.0. Then for each chunk, do:
    # chunk_max = max(acc)
    # exp_acc = exp(acc - chunk_max)
    # sum_exp += sum(exp_acc)
    # m_new = max(m, chunk_max)
    # We need a way to compute max and sum over vector acc. Triton has reductions. We can use tl.max(acc, axis=0) and tl.sum(exp_acc, axis=0). However, Triton reductions are designed for 2D tensors; for 1D, we can use tl.max and tl.sum.
    # To be robust, we'll use a temporary vector and mask: compute chunk_max = tl.max(acc, axis=0) and sum_exp += tl.sum(exp(acc - chunk_max), axis=0). m = m_new.

    # Start lse accumulation
    m = -float("inf")
    sum_exp = 0.0

    for t_base in range(0, L_tokens, BLOCK_L):
        t = t_base + tl.arange(0, BLOCK_L)
        mask_t = t < L_tokens

        # Load acc for this chunk from stored attn; but we cannot load acc here since we've already computed and stored attn. Instead, we'll recompute acc and update m/sum_exp.
        # Given the complexity, we'll implement a simpler: we'll recompute acc for this chunk (it's small overhead), and update m/sum_exp. For L_tokens small, this is fine.
        acc = tl.zeros([BLOCK_L], dtype=tl.float32)

        # Recompute Kc accumulation
        for k in range(0, Dc):
            qn_k = tl.load(qn_ptr + i * Dc + k, mask=True, other=0.0)
            Kc_chunk = tl.load(Kc_ptr + t * Dc + k, mask=mask_t, other=0.0)
            acc += qn_k * Kc_chunk

        # Recompute Kp accumulation
        for k in range(0, Dp):
            qp_k = tl.load(qp_ptr + i * Dp + k, mask=True, other=0.0)
            Kp_chunk = tl.load(Kp_ptr + t * Dp + k, mask=mask_t, other=0.0)
        # Note: Kp accumulation was lost. Fix by recomputing:
            Kp_chunk = tl.load(Kp_ptr + t * Dp + k, mask=mask_t, other=0.0)
            acc += qp_k * Kp_chunk

        # Scale
        acc = acc * sm_scale

        # Update lse stats
        # chunk_max = tl.max(acc, axis=0)
        # Triton reductions on vectors: tl.max(acc, axis=0) is not available; we can use tl.max(acc, axis=None) or tl.max(acc, axis=0) depending on Triton version.
        # Use tl.max(acc, axis=None)
        chunk_max = tl.max(acc, axis=None)
        # Compute sum_exp for this chunk: sum(exp(acc - chunk_max)) over valid t
        exp_acc = tl.exp(acc - chunk_max)
        # sum over chunk: Triton reduction over vector: tl.sum(exp_acc, axis=None)
        chunk_sum = tl.sum(exp_acc, axis=None)
        sum_exp += chunk_sum
        m = tl.maximum(m, chunk_max)

    # Compute final lse[i] = m + log(sum_exp) / ln(2)
    lse_val = m + math.log(sum_exp) / math.log(2.0)
    # Store lse
    tl.store(lse_ptr + i, lse_val)

# Kernel 2: matvec for attn @ Kc
@triton.jit
def matvec_kernel(
    attn_ptr, Kc_ptr, out_ptr,
    H: tl.constexpr, Dc: tl.constexpr, L: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    i = tl.program_id(0)  # head index
    # Compute out[i, :] = attn[i, :] @ Kc[:, :] where attn is [L], Kc is [L, Dc]
    out_vec = tl.zeros([Dc], dtype=tl.float32)
    for d_base in range(0, Dc, BLOCK_D):
        d = d_base + tl.arange(0, BLOCK_D)
        # Accumulate out_vec[d] += sum_{t=0..L-1} attn[i, t] * Kc[t, d]
        # Load Kc[t, d] chunk
        # Kc_ptr is 1D flattened: index = t * Dc + d
        # We need attn[i, t] for t in 0..L-1. attn_ptr is 1D with length H*L; offset = i * L + t
        acc_d = tl.zeros([BLOCK_D], dtype=tl.float32)
        for t in range(0, L):
            attn_t = tl.load(attn_ptr + i * L + t, mask=True, other=0.0)
            Kc_chunk = tl.load(Kc_ptr + t * Dc + d, mask=(d < Dc), other=0.0)
            acc_d += attn_t * Kc_chunk
        out_vec[d] = acc_d
    # Store out_vec to out[i, :]
    out_offset = i * Dc + d
    tl.store(out_ptr + out_offset, out_vec, mask=(d < Dc))


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Device and dtype checks
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA for Triton kernels."
        device = q_nope.device

        # Prepare constants
        batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages = ckv_cache.shape[0]
        assert num_qo_heads == 16, "num_qo_heads must be 16"
        assert head_dim_ckv == 512, "head_dim_ckv must be 512"
        assert head_dim_kpe == 64, "head_dim_kpe must be 64"
        assert ckv_cache.shape == (num_pages, 1, 512), "ckv_cache must have shape [num_pages, 1, 512]"
        assert kpe_cache.shape == (num_pages, 1, 64), "kpe_cache must have shape [num_pages, 1, 64]"
        assert kv_indptr.shape[0] == batch_size + 1, "kv_indptr length must be batch_size + 1"
        # Ensure inputs are contiguous and float32 for compute
        q_nope_f32 = q_nope.contiguous().to(torch.float32)  # [B, H, Dc]
        q_pe_f32 = q_pe.contiguous().to(torch.float32)      # [B, H, Dp]
        # Squeeze the size-1 dims
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, Dc]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, Dp]

        # Output buffers
        output = torch.zeros((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        # For each batch element
        for b in range(batch_size):
            # Compute number of tokens for this batch element
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L_tokens = page_end - page_beg
            # If no tokens, set output[b] zero and lse[b] = -inf
            if L_tokens <= 0:
                continue

            # Gather token indices
            tok_idx = kv_indices[page_beg:page_end].contiguous().to(torch.int32)
            # Gather Kc/Kp rows for these tokens (num_tokens x Dc) and (num_tokens x Dp)
            Kc = Kc_all[tok_idx]  # [L_tokens, Dc], float32
            Kp = Kp_all[tok_idx]  # [L_tokens, Dp], float32

            # Prepare flattened q vectors for Triton: [H, Dc] and [H, Dp]
            # We need to pass q_nope[b] and q_pe[b] as 1D flattened pointers
            qn_flat = q_nope_f32[b].contiguous().view(-1)  # [H*Dc] -> but H=16, Dc=512, so 8192
            qp_flat = q_pe_f32[b].contiguous().view(-1)    # [H*Dp] -> 16*64 = 1024

            # Allocate attn buffer [H, L_tokens] as 1D: length = H * L_tokens
            attn = torch.empty((num_qo_heads, L_tokens), dtype=torch.float32, device=device).contiguous()  # [H, L]
            # Launch kernel to compute attn and lse per head
            # Grid: one program per head
            grid = (num_qo_heads,)
            compute_logits_scaled_and_attn_kernel[grid](
                qn_flat, qp_flat, Kc.view(-1), Kp.view(-1),
                attn.view(-1), lse[b].view(1),  # lse[b] is scalar float32
                H=num_qo_heads, Dc=head_dim_ckv, Dp=head_dim_kpe,
                L_tokens=L_tokens, sm_scale=sm_scale,
                BLOCK_L=128,  # chunk size over tokens
            )

            # Compute out_vec for each head i: attn[i] @ Kc
            # We need out[i, :] where out is [H, Dc]
            out_vec = torch.empty((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)  # [H, Dc]
            # Launch matvec kernel: grid over heads
            matvec_kernel[grid](
                attn.view(-1), Kc.view(-1), out_vec.view(-1),
                H=num_qo_heads, Dc=head_dim_ckv, L=L_tokens,
                BLOCK_D=128,  # chunk size over Dc
            )

            # Store to output[b, i] as bfloat16
            for i in range(num_qo_heads):
                output[b, i] = out_vec[i].to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
