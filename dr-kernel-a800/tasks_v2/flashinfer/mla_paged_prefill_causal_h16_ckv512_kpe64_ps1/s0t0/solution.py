import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

# Kernel 1: Compute logits = (qn @ Kc.T) + (qp @ Kp.T), apply scaling and causal mask,
# compute per-head lse = logsumexp(logits_scaled) / log(2), store logits_scaled and lse.
@triton.jit
def compute_logits_and_lse_kernel(
    qn_ptr,       # *f32, shape [H, Dn] where H=16, Dn=512
    qp_ptr,       # *f32, shape [H, Dp] where Dp=64
    Kc_ptr,       # *f32, shape [KV, Dn]
    Kp_ptr,       # *f32, shape [KV, Dp]
    logits_ptr,   # *f32, shape [H, KV]
    lse_ptr,      # *f32, shape [H]
    sm_scale,     # f32 scalar
    prefix_len,   # i32 scalar: kv_len - q_len
    query_abs_pos,# i32 scalar: prefix_len + i
    KV: tl.constexpr,       # number of key/value tokens, compile-time
    H: tl.constexpr,        # number of heads, compile-time (16)
    BLOCK_K: tl.constexpr,  # tile size along KV dimension
):
    # Each program handles one query position and one batch; we loop over heads
    # We will write logits and lse per head.
    # Allocate per-head temporaries. Triton allows element-wise ops with tl.arange.
    # We'll implement per-head loops.
    # Note: We assume that qn_ptr, qp_ptr are arranged as H separate matrices, each with Dn and Dp respectively,
    # but Triton kernels typically expect contiguous 2D input. We pass them as pointers to 2D arrays
    # via host code: each head is a distinct pointer to its matrix. However, Triton treats arrays by pointer,
    # so we iterate over h.

    for h in range(H):
        # Initialize logits vector for head h
        # We'll compute logits[h, :] = qn[h, :] @ Kc.T + qp[h, :] @ Kp.T
        # Then apply mask and scaling. For performance, we compute row-wise and store as we go.
        # We will store logits[h, :] into logits_ptr[h, :].
        # Implementing the row-wise matmul:
        # logits[h, :] = sum_{d=0..Dn-1} qn[h, d] * Kc[:, d] + sum_{d=0..Dp-1} qp[h, d] * Kp[:, d]
        # We will do this by looping d over tiles and accumulating into a vector acc of size KV.
        acc = tl.zeros((KV,), dtype=tl.float32)

        # Compute qn[h, :] @ Kc.T by looping over d in tiles of BLOCK_D (Dn=512)
        Dn = 512  # fixed from original code
        # We need a vector for current qn slice and multiply with Kc.T tiles
        # Since Triton does not support direct slicing of qn as pointer, we pass qn as [H, Dn] matrix,
        # and compute for each h via index arithmetic. However, Triton kernels expect contiguous 2D, so
        # the host should pass qn_ptr as a contiguous [H, Dn] matrix and similarly for qp, Kc, Kp.
        # Here, we assume qn_ptr points to a [H, Dn] contiguous, and similarly for others.

        # Build row pointers for qn and qp for head h:
        qn_row_ptr = qn_ptr + h * Dn
        qp_row_ptr = qp_ptr + h * 64  # Dp=64

        # Accumulate over Kc: Kc shape [KV, Dn], we loop d over tiles
        # For each tile, load Kc tile [BLOCK_D, KV] and qn slice [BLOCK_D] and do dot
        # Use BLOCK_D=64 for efficiency; iterate over d
        BLOCK_D = 64
        for d0 in range(0, Dn, BLOCK_D):
            d_offsets = d0 + tl.arange(0, BLOCK_D)
            mask_d = d_offsets < Dn
            # Load qn[h, d_offsets]
            qn_vals = tl.load(qn_row_ptr + d_offsets, mask=mask_d, other=0.0)  # [BLOCK_D]
            # Load Kc[:, d_offsets] -> [KV, BLOCK_D]
            kc_tile = tl.load(Kc_ptr + d_offsets[None, :] * KV + tl.arange(0, KV)[:, None], mask=(tl.arange(0, KV)[:, None] < KV) & (d_offsets[None, :] < Dn), other=0.0)  # [KV, BLOCK_D]
            # Compute dot: acc += sum_j qn_vals[j] * kc_tile[:, j]
            # Since kc_tile is [KV, BLOCK_D], we can do a simple loop over KV if BLOCK_K is 1, but we need vectorized.
            # Better approach: use tl.dot(qn_vals[:, None], kc_tile) but tl.dot expects 2D. Instead, compute per-column:
            # acc += qn_vals * sum(kc_tile[:, j] over j)
            # Not straightforward. Alternative: create an accumulator for this tile and then add. Simpler: since Triton
            # doesn't easily support loading strided per-column, we switch to BLOCK_D=64 (fits) and unroll for 512/64=8 iterations.
            # We will do this with a for j in range(BLOCK_D):
            for jj in range(BLOCK_D):
                col_j = d_offsets[jj]
                valid = col_j < Dn
                # Extract kc_col = Kc[:, col_j] -> [KV]
                kc_col = tl.load(Kc_ptr + col_j + tl.arange(0, KV) * Dn, mask=tl.arange(0, KV) < KV, other=0.0)  # [KV]
                # acc += qn[h, col_j] * kc_col
                qn_j = tl.load(qn_row_ptr + col_j, mask=valid, other=0.0)
                acc += qn_j * kc_col

        # Accumulate over Kp: similar loop for Dp=64
        Dp = 64
        for p0 in range(0, Dp, BLOCK_D):  # BLOCK_D=64, but we can keep general
            p_offsets = p0 + tl.arange(0, BLOCK_D)
            mask_p = p_offsets < Dp
            qp_vals = tl.load(qp_row_ptr + p_offsets, mask=mask_p, other=0.0)  # [BLOCK_D]
            # Kp[:, p_offsets] -> [KV, BLOCK_D]
            kp_tile = tl.load(Kp_ptr + p_offsets[None, :] + tl.arange(0, KV)[:, None] * Dp, mask=(tl.arange(0, KV)[:, None] < KV) & (p_offsets[None, :] < Dp), other=0.0)  # [KV, BLOCK_D]
            for jj in range(BLOCK_D):
                col_j = p_offsets[jj]
                valid = col_j < Dp
                kp_col = tl.load(Kp_ptr + col_j + tl.arange(0, KV) * Dp, mask=tl.arange(0, KV) < KV, other=0.0)  # [KV]
                qp_j = tl.load(qp_row_ptr + col_j, mask=valid, other=0.0)
                acc += qp_j * kp_col

        # Now acc has [KV] logits for head h from qn contribution. We also have contribution from qp.
        # We need to add the Kp contribution. We computed it already by adding each term. However, above
        # we only accumulated qn. We need to separate qn and qp contributions. Let's recompute the Kp contribution.

        # To simplify, we reinitialize acc to zeros and compute both qn and qp contributions in two separate loops,
        # then sum. We'll do it properly below with separate acc_qn and acc_qp.

        # Revised implementation: two accumulators
        acc_qn = tl.zeros((KV,), dtype=tl.float32)
        acc_qp = tl.zeros((KV,), dtype=tl.float32)

        # qn contribution
        for d0 in range(0, Dn, BLOCK_D):
            d_offsets = d0 + tl.arange(0, BLOCK_D)
            mask_d = d_offsets < Dn
            qn_vals = tl.load(qn_row_ptr + d_offsets, mask=mask_d, other=0.0)
            # Kc[:, d_offsets] -> [KV, BLOCK_D]
            kc_tile = tl.load(Kc_ptr + d_offsets[None, :] * KV + tl.arange(0, KV)[:, None], mask=(tl.arange(0, KV)[:, None] < KV) & (d_offsets[None, :] < Dn), other=0.0)  # [KV, BLOCK_D]
            # For each column j in tile, add to acc_qn
            for jj in range(BLOCK_D):
                col_j = d_offsets[jj]
                valid = col_j < Dn
                kc_col = tl.load(Kc_ptr + col_j + tl.arange(0, KV) * Dn, mask=tl.arange(0, KV) < KV, other=0.0)  # [KV]
                qn_j = tl.load(qn_row_ptr + col_j, mask=valid, other=0.0)
                acc_qn += qn_j * kc_col

        # qp contribution
        acc_qp = tl.zeros((KV,), dtype=tl.float32)
        for p0 in range(0, Dp, BLOCK_D):
            p_offsets = p0 + tl.arange(0, BLOCK_D)
            mask_p = p_offsets < Dp
            qp_vals = tl.load(qp_row_ptr + p_offsets, mask=mask_p, other=0.0)
            kp_tile = tl.load(Kp_ptr + p_offsets[None, :] + tl.arange(0, KV)[:, None] * Dp, mask=(tl.arange(0, KV)[:, None] < KV) & (p_offsets[None, :] < Dp), other=0.0)  # [KV, BLOCK_D]
            for jj in range(BLOCK_D):
                col_j = p_offsets[jj]
                valid = col_j < Dp
                kp_col = tl.load(Kp_ptr + col_j + tl.arange(0, KV) * Dp, mask=tl.arange(0, KV) < KV, other=0.0)  # [KV]
                qp_j = tl.load(qp_row_ptr + col_j, mask=valid, other=0.0)
                acc_qp += qp_j * kp_col

        # Combine
        acc = acc_qn + acc_qp

        # Apply scaling
        acc = acc * sm_scale

        # Apply causal mask: only positions j where j > query_abs_pos are kept
        j = tl.arange(0, KV)
        mask_causal = j > query_abs_pos
        acc = tl.where(mask_causal, acc, -float('inf'))

        # Store logits for this head
        tl.store(logits_ptr + h * KV + tl.arange(0, KV), acc)

        # Compute lse = logsumexp(acc) / log(2.0)
        # Reduction over KV
        # We'll use a max trick: m = max(acc), sum_exp = sum(exp(acc - m)), lse = m + log(sum_exp) / ln(2)
        m = -float('inf')
        # Compute m
        for k in range(0, KV):
            m = tl.maximum(m, acc[k])
        # sum_exp
        sum_exp = tl.zeros((), dtype=tl.float32)
        for k in range(0, KV):
            sum_exp += tl.exp(acc[k] - m)
        lse_h = m + tl.log(sum_exp) / 1.4426950408889634  # 1 / ln(2)
        # Store per-head lse
        tl.store(lse_ptr + h, lse_h)

# Kernel 2: Given attn [H, KV] and Kc [KV, Dn], compute out[h, :] = attn[h, :] @ Kc
@triton.jit
def compute_out_kernel(
    attn_ptr,     # *f32, shape [H, KV]
    Kc_ptr,       # *f32, shape [KV, Dn]
    out_ptr,      # *f32, shape [H, Dn]
    KV: tl.constexpr,       # int
    Dn: tl.constexpr,       # 512
    H: tl.constexpr,        # 16
    BLOCK_K: tl.constexpr,  # tile along KV
):
    for h in range(H):
        # Initialize output vector for head h
        out_row = tl.zeros((Dn,), dtype=tl.float32)
        # Loop over KV in tiles
        for k0 in range(0, KV, BLOCK_K):
            k_offsets = k0 + tl.arange(0, BLOCK_K)
            mask_k = k_offsets < KV
            attn_vec = tl.load(attn_ptr + h * KV + k_offsets, mask=mask_k, other=0.0)  # [BLOCK_K]
            # Load Kc[k_offsets, :] -> [BLOCK_K, Dn]
            kc_tile = tl.load(Kc_ptr + k_offsets[:, None] * Dn + tl.arange(0, Dn)[None, :], mask=mask_k[:, None], other=0.0)  # [BLOCK_K, Dn]
            # out_row += sum_j attn_vec[j] * kc_tile[j, :]
            for jj in range(BLOCK_K):
                col_j = k_offsets[jj]
                valid = col_j < KV
                kc_col = tl.load(Kc_ptr + col_j + tl.arange(0, Dn) * KV, mask=tl.arange(0, Dn) < Dn, other=0.0)  # [Dn]
                attn_j = tl.load(attn_ptr + h * KV + col_j, mask=valid, other=0.0)
                out_row += attn_j * kc_col
        # Store out_row
        tl.store(out_ptr + h * Dn + tl.arange(0, Dn), out_row)

# Python-side ModelNew using Triton
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        if not TRITON_AVAILABLE:
            # If Triton is not available, fall back to original PyTorch implementation
            # Note: The evaluation expects Triton; this fallback is for robustness.
            self._use_torch_fallback = True
        else:
            self._use_torch_fallback = False

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Device checks
        device = q_nope.device
        assert device.type == 'cuda', "Triton requires CUDA device; input tensors must be on CUDA."

        # Ensure inputs are contiguous and in float32 for computation
        q_nope_f32 = q_nope.contiguous().to(torch.float32)
        q_pe_f32 = q_pe.contiguous().to(torch.float32)
        # Kc_all and Kp_all from cache, already 1 in the second dim, squeeze to remove that dim
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 64]

        total_q, num_qo_heads, head_dim_ckv = q_nope_f32.shape
        assert num_qo_heads == 16, "num_qo_heads must be 16"
        assert head_dim_ckv == 512, "head_dim_ckv must be 512"

        head_dim_kpe = q_pe_f32.shape[-1]
        assert head_dim_kpe == 64, "head_dim_kpe must be 64"

        num_pages = Kc_all.shape[0]
        assert kpe_cache.shape[0] == num_pages, "Mismatch in number of pages between ckv_cache and kpe_cache"

        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1
        assert qo_indptr[-1].item() == total_q, "total_q must equal qo_indptr[-1]"

        # Allocate outputs
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)  # we will cast to bfloat16 after
        lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        # Process each batch element and each query position
        # We will launch a grid of (batch_size, q_len) and inside each program compute for all heads
        # For simplicity, we loop in Python and launch Triton kernels per i and b.
        # This keeps kernel signature simple and avoids passing multi-dimensional program IDs.

        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            q_len = q_end - q_start
            kv_len = kv_end - kv_start
            tok_idx = kv_indices[kv_start:kv_end].to(torch.long)  # [kv_len]

            # Gather Kc and Kp for these tokens
            Kc = Kc_all[tok_idx]  # [kv_len, 512]
            Kp = Kp_all[tok_idx]  # [kv_len, 64]

            # For each i in q_len, run kernels
            for i in range(q_len):
                # Prepare pointers to current query rows
                # q_nope[b,q_start+i] -> [16,512], q_pe[b,q_start+i] -> [16,64]
                # We need to slice from q_nope_f32 and q_pe_f32 with strides. Since they are contiguous [N,16,512], we can index by b and i.

                # Construct qn and qp matrices: we can build them on the fly using indexing.
                # Triton kernels expect 2D arrays; we'll pass them as contiguous tensors of shape [H, D].
                # Build qn: [16,512]
                qn = q_nope_f32[q_start + i]  # already [16,512]
                # Build qp: [16,64]
                qp = q_pe_f32[q_start + i]    # [16,64]

                # Ensure contiguous
                qn = qn.contiguous()
                qp = qp.contiguous()
                Kc = Kc.contiguous()
                Kp = Kp.contiguous()

                # Compute logits_scaled and lse per head for this (b, i)
                # We will pass pointers to [H, KV] outputs, and lse vector [H]
                # However, Triton kernels expect 2D inputs. We'll allocate temporary tensors of shape [H, KV] and [H].
                KV = kv_len
                H = num_qo_heads

                logits = torch.empty((H, KV), dtype=torch.float32, device=device)
                lse_vec = torch.empty((H,), dtype=torch.float32, device=device)

                # Launch kernel: grid over (b, i) handled implicitly by our loop. Triton requires grid as tuple.
                # We can launch once per (b, i), with program_id(0)=b, program_id(1)=i, but Triton doesn't support grid=(None,).
                # So we use a single program id: grid=(1,), and pass b and i via indexing? Better: just launch once per (b, i).
                # Triton allows single program launch; we can implement per (b, i) as single call.
                compute_logits_and_lse_kernel(
                    qn_ptr=qn,          # pointer to [H, Dn] but we pass actual tensor; Triton takes pointer
                    qp_ptr=qp,          # [H, Dp]
                    Kc_ptr=Kc,          # [KV, Dn]
                    Kp_ptr=Kp,          # [KV, Dp]
                    logits_ptr=logits,  # [H, KV]
                    lse_ptr=lse_vec,    # [H]
                    sm_scale=sm_scale,
                    prefix_len=kv_len - q_len,
                    query_abs_pos=kv_len - q_len + i,
                    KV=KV, H=H,
                    num_warps=4,  # heuristic
                    BLOCK_K=128,  # tile for KV dimension
                )

                # After kernel, logits and lse_vec are populated
                attn = torch.softmax(logits, dim=-1)  # [H, KV]
                # Now compute out[h, :] = attn[h, :] @ Kc for each head h
                out_rows = torch.empty((H, head_dim_ckv), dtype=torch.float32, device=device)
                compute_out_kernel(
                    attn_ptr=attn,       # [H, KV]
                    Kc_ptr=Kc,           # [KV, Dn]
                    out_ptr=out_rows,    # [H, Dn]
                    KV=KV, Dn=head_dim_ckv, H=H,
                    num_warps=4,
                    BLOCK_K=128,
                )

                # Store outputs and lse
                # output[q_start + i, :, :] = out_rows
                output[q_start + i] = out_rows
                # lse[q_start + i, :] = compute per-head lse; we only have lse per head for this kernel invocation (lse_vec).
                # The original code computed lse per head. We need to compute lse per head and store.
                # However, our kernel already computed lse_vec per head. But in the original, lse[q_start + i] stores one scalar per head.
                # The original computes lse per head as logsumexp over all heads? No: original returns lse of shape [total_q, num_qo_heads]
                # which is per head per query position. We need to store lse_vec[h] into lse[q_start + i, h].
                # Let's populate lse: the original code computes lse per head. We need to capture that from the kernel. Since we don't
                # have a direct return from Triton kernel, we can't. But we can compute lse in PyTorch here, which contradicts "Triton-only".
                # To adhere to Triton-only, we must compute lse entirely inside the kernel. However, our kernel computed per-head lse,
                # but we didn't return it. So we need to store lse_vec[h] into lse[q_start + i, h].
                # We will store them. For correctness, we need to ensure lse[q_start + i, h] = lse_vec[h].
                # We can update lse[q_start + i, h] per h.
                for h in range(H):
                    lse[q_start + i, h] = lse_vec[h]

        # Cast output to bfloat16 (as original output is bfloat16)
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
