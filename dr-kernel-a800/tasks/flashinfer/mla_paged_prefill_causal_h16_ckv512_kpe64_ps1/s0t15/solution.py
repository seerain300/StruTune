import math
import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


# Kernel 1: Compute S[h, :] = qn_row[h, :] @ Kc.T where qn_row is [Dn], Kc is [KV, Dn], S is [KV]
# We pass qn_ptr as a pointer to a [H, Dn] matrix; in practice, we set H=1 and point to the row.
@triton.jit
def matmul_qn_kc_kernel(
    qn_ptr,             # *float32, points to [H, Dn]; we will pass pointer to a single row as H=1
    Kc_ptr,             # *float32, points to [KV, Dn]
    logits_ptr,         # *float32, output [KV]
    H: tl.int32,        # number of heads (we pass 1)
    Dn: tl.constexpr,   # 512
    KV: tl.constexpr,   # number of KV tokens
    num_warps: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program handles one row h. We pass H=1, so we directly compute for that row.
    h = 0  # since H==1, we can ignore h loop
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Iterate over K dimension in tiles
    for k0 in range(0, KV, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        # Load qn_row[h, :] in chunks of BLOCK_N. Since h=0, we load qn_ptr[0, :].
        qn_vec = tl.load(qn_ptr + 0 * Dn + tl.arange(0, Dn))  # [Dn]
        # Kc tile: [BLOCK_K, Dn]
        Kc_tile = tl.load(Kc_ptr + k_idx[:, None] * Dn + tl.arange(0, Dn)[None, :], mask=(k_idx[:, None] < KV))
        # acc += sum over K tile: (qn_vec * Kc_tile).sum(axis=1)
        # We need to multiply qn_vec by each column of Kc_tile and reduce. Implement as:
        # Since qn_vec is [Dn], Kc_tile is [BLOCK_K, Dn], we want acc[j] += sum_{kk} Kc_tile[kk, j] * qn_vec[j]
        # Note: Triton does not support direct matmul here; we implement reduction explicitly.
        # We will load Kc columns for each j in BLOCK_N.
        # For simplicity and correctness, we use tl.dot(qn_vec, Kc_tile.T) which yields [BLOCK_K].
        # However, tl.dot requires compatible dims. Instead, do explicit per-j loop over BLOCK_N.
        # We'll set BLOCK_N = Dn and loop j across Dn.
        # But to keep vectorized: we compute acc for all columns j in one vector by using:
        # acc_j += sum_k Kc_tile[k, j] * qn_vec[j]
        # We can write a small loop over kkk within BLOCK_K and update acc_j for all j. Triton allows loops.
        for jj in range(0, Dn):
            # Accumulate scalar over K tiles
            acc_j_scalar = 0.0
            for kk in range(0, BLOCK_K):
                k_valid = k0 + kk < KV
                # Load Kc[k0+kk, jj]
                kc_val = tl.load(Kc_ptr + (k0 + kk) * Dn + jj, mask=k_valid, other=0.0)
                # qn[jj]
                qn_j = tl.load(qn_ptr + 0 * Dn + jj)
                acc_j_scalar += kc_val * qn_j
            # Place acc_j into acc vector at position jj
            acc[jj] = acc_j_scalar
        # Store logits: we store acc (the full vector) into logits_ptr
        # To store a vector, we write per element with masks. We can write across KV positions by constructing
        # the positions j in vector form. But since we accumulated acc over j loop above, we need to store
        # to logits_ptr for j in [0, KV). We can write only the first Dn elements. This kernel is intended
        # to compute per-row S vector. For simplicity, we assume KV <= Dn and store acc[0:KV].
        # However, to be general, we need a 1D output of size KV. We'll instead compute per-K element and
        # write directly. The best approach is to compute S[j] in a separate loop over j, but Triton doesn't
        # support direct storing of arbitrary positions without scatter. Therefore, we implement S computation
        # in a separate kernel that writes to a 1D output using scalar j.
        # Given constraints in this environment, we switch to a simpler kernel: compute S vector by looping j
        # and loading columns from Kc_ptr. We'll redefine with simpler 1D accumulation.

    # The above complex construction is incorrect in Triton. We should instead implement a scalar-loop per j:
    # Here, we simplify by computing S[j] directly and storing into logits_ptr[j].
    for j in range(0, KV):
        acc_j = 0.0
        for kk in range(0, Dn):
            qn_j = tl.load(qn_ptr + 0 * Dn + kk)
            kc_j = tl.load(Kc_ptr + j * Dn + kk)
            acc_j += qn_j * kc_j
        # Apply scaling here if needed. For now, we store acc_j (we'll scale in host). We can store acc_j.
        tl.store(logits_ptr + j, acc_j)

# Simplified kernel: compute S[h, :] directly and store to logits_ptr
@triton.jit
def compute_qn_kc_vec_kernel(
    qn_ptr,        # *float32, points to qn_row[h, :]
    Kc_ptr,        # *float32, points to [KV, Dn]
    logits_ptr,    # *float32, output [KV]
    Dn: tl.constexpr,
    KV: tl.constexpr,
):
    h = 0  # single head
    for j in range(0, KV):
        acc_j = 0.0
        for kk in range(0, Dn):
            qn_j = tl.load(qn_ptr + kk)
            kc_j = tl.load(Kc_ptr + j * Dn + kk)
            acc_j += qn_j * kc_j
        tl.store(logits_ptr + j, acc_j)

# Kernel 2: Compute T[h, :] = qp_row[h, :] @ Kp.T where qp_row is [Dp], Kp is [KV, Dp], T is [KV]
@triton.jit
def compute_qp_kp_vec_kernel(
    qp_ptr,        # *float32, points to qp_row[h, :]
    Kp_ptr,        # *float32, points to [KV, Dp]
    logits_ptr,    # *float32, output [KV]
    Dp: tl.constexpr,
    KV: tl.constexpr,
):
    h = 0
    for j in range(0, KV):
        acc_j = 0.0
        for kk in range(0, Dp):
            qp_j = tl.load(qp_ptr + kk)
            kp_j = tl.load(Kp_ptr + j * Dp + kk)
            acc_j += qp_j * kp_j
        tl.store(logits_ptr + j, acc_j)

# Kernel 3: Compute lse for one head h: lse[h] = logsumexp(logits_scaled[h, :]) / ln(2)
@triton.jit
def lse_row_kernel(
    logits_ptr,     # *float32, [KV]
    lse_ptr,        # *float32, scalar output
    KV: tl.constexpr,
    SM_SCALE: tl.float32,
    CAUSAL_POS: tl.int32,
):
    h = 0
    # Load logits vector
    logits = tl.load(logits_ptr + tl.arange(0, KV))
    # Apply scaling
    logits = logits * SM_SCALE
    # Apply causal mask: positions j <= CAUSAL_POS -> -inf
    j = tl.arange(0, KV)
    mask = j > CAUSAL_POS
    logits = tl.where(mask, logits, -float('inf'))
    # Compute max
    max_val = tl.max(logits)
    # Subtract max
    logits_shift = logits - max_val
    # Compute sum(exp)
    exp_sum = tl.sum(tl.exp(logits_shift))
    # lse = log(sum) / ln(2)
    ln2 = 1.4426950408889634  # 1 / log(2)
    lse_val = tl.log(exp_sum) / ln2
    # Store lse[h]
    tl.store(lse_ptr, lse_val)

# Kernel 4: Compute softmax for one head h: attn[h, :] = softmax(logits_scaled)
@triton.jit
def softmax_row_kernel(
    logits_ptr,     # *float32, [KV]
    attn_ptr,       # *float32, [KV]
    KV: tl.constexpr,
    SM_SCALE: tl.float32,
    CAUSAL_POS: tl.int32,
):
    h = 0
    logits = tl.load(logits_ptr + tl.arange(0, KV))
    logits = logits * SM_SCALE
    j = tl.arange(0, KV)
    mask = j > CAUSAL_POS
    logits = tl.where(mask, logits, -float('inf'))
    max_val = tl.max(logits)
    logits_shift = logits - max_val
    exp_vec = tl.exp(logits_shift)
    sum_exp = tl.sum(exp_vec)
    attn_vec = exp_vec / sum_exp
    tl.store(attn_ptr + tl.arange(0, KV), attn_vec)

# Kernel 5: out[h, :] = attn[h, :] @ Kc
@triton.jit
def matmul_attn_kc_kernel(
    attn_ptr,       # *float32, [KV]
    Kc_ptr,         # *float32, [KV, Dn]
    out_ptr,        # *float32, [Dn]
    Dn: tl.constexpr,
    KV: tl.constexpr,
):
    h = 0
    acc = tl.zeros((Dn,), dtype=tl.float32)
    for j in range(0, KV):
        attn_j = tl.load(attn_ptr + j)
        kc_row = tl.load(Kc_ptr + j * Dn + tl.arange(0, Dn))
        acc += attn_j * kc_row
    tl.store(out_ptr + tl.arange(0, Dn), acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA"
        device = q_nope.device

        # Constants
        total_q = int(qo_indptr[-1].item())
        num_qo_heads = 16
        head_dim_ckv = 512
        head_dim_kpe = 64

        # Prepare Kc_all and Kp_all (cached values)
        # Note: original asserts assume shapes; we follow the logic but keep assertions relaxed here.
        Kc_all = ckv_cache.to(torch.float32)
        Kp_all = kpe_cache.to(torch.float32)

        # Allocate output and lse
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)  # will cast to bfloat16
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Number of batches
        B = qo_indptr.numel() - 1

        # Process each batch element
        for b in range(B):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            if q_start >= q_end or kv_start >= kv_end:
                continue
            q_len = q_end - q_start
            kv_len = kv_end - kv_start

            # Gather Kc and Kp for this batch
            tok_idx = kv_indices[kv_start:kv_end].to(torch.int32).to(device)  # indices already int32
            Kc = Kc_all[tok_idx]  # [kv_len, 512]
            Kp = Kp_all[tok_idx]  # [kv_len, 64]

            # Loop over queries i in this batch
            for i in range(q_len):
                # Compute per-head results
                for h in range(num_qo_heads):
                    # Load qn_row and qp_row
                    qn_row = q_nope[q_start + i, h, :].to(torch.float32).to(device)  # [512]
                    qp_row = q_pe[q_start + i, h, :].to(torch.float32).to(device)   # [64]

                    # Output row tensor (will be filled by Triton)
                    out_row = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)

                    # Compute S = qn_row @ Kc.T using Triton kernel (we'll implement as vectorized per-j)
                    # Allocate S_vec
                    S_vec = torch.empty((kv_len,), dtype=torch.float32, device=device)
                    # Launch compute_qn_kc_vec_kernel: Triton expects pointers; this kernel writes S_vec
                    # We pass qn_row as a 1D vector pointer.
                    qn_ptr = qn_row  # Triton will read as 1D
                    Kc_ptr = Kc
                    S_kernel = compute_qn_kc_vec_kernel[(1,)](
                        qn_ptr, Kc_ptr, S_vec,
                        Dn=512, KV=kv_len, num_warps=1, BLOCK_K=64
                    )

                    # Compute T = qp_row @ Kp.T using Triton kernel
                    T_vec = torch.empty((kv_len,), dtype=torch.float32, device=device)
                    Kp_ptr = Kp
                    T_kernel = compute_qp_kp_vec_kernel[(1,)](
                        qp_row, Kp_ptr, T_vec,
                        Dp=64, KV=kv_len, num_warps=1, BLOCK_K=64
                    )

                    # Sum and scale
                    logits = S_vec + T_vec  # [KV]
                    logits_scaled = logits * sm_scale

                    # Compute lse[h] using Triton
                    lse_pos = torch.empty((1,), dtype=torch.float32, device=device)
                    lse_kernel = lse_row_kernel[(1,)](
                        logits_scaled, lse_pos,
                        KV=kv_len, SM_SCALE=sm_scale, CAUSAL_POS=(kv_len - q_len + i)
                    )
                    lse[q_start + i, h] = lse_pos[0]

                    # Compute attn[h, :] using Triton
                    attn_vec = torch.empty((kv_len,), dtype=torch.float32, device=device)
                    softmax_kernel = softmax_row_kernel[(1,)](
                        logits_scaled, attn_vec,
                        KV=kv_len, SM_SCALE=sm_scale, CAUSAL_POS=(kv_len - q_len + i)
                    )

                    # Compute out[h, :] = attn_vec @ Kc using Triton
                    out_kernel = matmul_attn_kc_kernel[(1,)](
                        attn_vec, Kc, out_row,
                        Dn=512, KV=kv_len, num_warps=1
                    )

                    # Store outputs
                    output[q_start + i, h, :] = out_row

        # Return output and lse, with original dtypes
        output = output.to(torch.bfloat16)
        lse = lse  # already float32
        return output, lse


def run(*args):
    return ModelNew()(*args)
