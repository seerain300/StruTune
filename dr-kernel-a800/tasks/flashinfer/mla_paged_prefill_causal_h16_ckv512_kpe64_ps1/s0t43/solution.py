import torch
import triton
import triton.language as tl

# Triton kernel: compute logits for a single (b, i, h)
# Inputs:
#   q_nope_ptr: [N, 16, 512] float32
#   q_pe_ptr: [N, 16, 64] float32
#   Kc_ptr: [M, 512] float32 (already gathered per batch)
#   Kp_ptr: [M, 64] float32 (already gathered per batch)
#   qo_indptr_ptr: [len_indptr] int32, where len_indptr >= 2 and qo_indptr[-1] == total_q
#   kv_indptr_ptr: [len_indptr] int32
#   kv_indices_ptr: [num_kv_indices] int32 (indices into cache)
#   sm_scale: float32 scalar
#   b: int32 batch index
#   i: int32 query index within batch b (q_start + i)
#   h: int32 head index (0..15)
# Outputs:
#   logits_ptr: [KV] float32 (logits for head h after masking and scaling)
@triton.jit
def compute_logits_kernel(
    q_nope_ptr, q_pe_ptr, Kc_ptr, Kp_ptr,
    qo_indptr_ptr, kv_indptr_ptr, kv_indices_ptr,
    sm_scale: tl.float32,
    b: tl.int32, i: tl.int32, h: tl.int32,
    total_q: tl.int32,
    KV: tl.int32,  # number of KV tokens for this batch element
    Dn: tl.constexpr,  # 512
    Dp: tl.constexpr,  # 64
    BLOCK_J: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    # Load qn_row[h, :] and qp_row[h, :]
    # q_nope layout: [N, 16, 512], we address (qo_indptr[b+1]-1)th row of batch b
    q_start = tl.load(qo_indptr_ptr + b)  # int32
    q_row = q_start + i
    qn_row = tl.load(q_nope_ptr + q_row * 16 * 512 + h * 512)  # [512]
    qp_row = tl.load(q_pe_ptr + q_row * 16 * 64 + h * 64)      # [64]

    # Prepare logits vector
    logits = tl.zeros([KV], dtype=tl.float32)

    # Accumulate qn_row @ Kc.T over K tiles
    # Kc layout: [KV, 512]
    for j in range(0, Dn, BLOCK_J):
        offs_j = j + tl.arange(0, BLOCK_J)
        mask_j = offs_j < Dn
        acc_qn = tl.zeros([BLOCK_J], dtype=tl.float32)
        # dot(qn_row, Kc[:, j:j+BLOCK_J])
        for k in range(0, Dn):
            kc_k = tl.load(Kc_ptr + k * Dn + offs_j, mask=mask_j, other=0.0)
            acc_qn += qn_row[k] * kc_k
        # add acc_qn to logits
        logits += tl.where(mask_j, acc_qn, 0.0)

    # Accumulate qp_row @ Kp.T over K tiles
    # Kp layout: [KV, 64]
    for j in range(0, Dp, BLOCK_J):
        offs_j = j + tl.arange(0, BLOCK_J)
        mask_j = offs_j < Dp
        acc_qp = tl.zeros([BLOCK_J], dtype=tl.float32)
        # dot(qp_row, Kp[:, j:j+BLOCK_J])
        for k in range(0, Dp):
            kp_k = tl.load(Kp_ptr + k * Dp + offs_j, mask=mask_j, other=0.0)
            acc_qp += qp_row[k] * kp_k
        # add acc_qp to logits
        logits += tl.where(mask_j, acc_qp, 0.0)

    # Scale
    logits *= sm_scale

    # Apply causal mask: for each j, keep logits[j] if j > (KV - (q_end - q_start) + i), else -inf
    # prefix_len = KV - q_len; j > prefix_len + i
    q_len = tl.load(qo_indptr_ptr + b + 1) - q_start
    prefix_len = KV - q_len
    cond = (tl.arange(0, KV) > (prefix_len + i))
    logits = tl.where(cond, logits, -float('inf'))

    # Store logits
    tl.store(logits_ptr, logits)


# Triton kernel: compute logsumexp over masked logits
@triton.jit
def lse_row_kernel(
    logits_ptr, lse_ptr, KV: tl.int32, inv_ln2: tl.float32
):
    # Numerically stable: lse = log(sum(exp(x - max))) / ln(2)
    x = tl.load(logits_ptr)
    max_x = tl.max(x, axis=0)
    sum_exp = tl.sum(tl.exp(x - max_x), axis=0)
    lse = tl.log(sum_exp) * inv_ln2
    tl.store(lse_ptr, lse)


# Triton kernel: compute softmax over masked logits (masked positions become 0 in exp)
@triton.jit
def softmax_row_kernel(
    logits_ptr, attn_ptr, KV: tl.int32
):
    x = tl.load(logits_ptr)
    max_x = tl.max(x, axis=0)
    exp_x = tl.exp(x - max_x)
    exp_x = tl.where(x == -float('inf'), 0.0, exp_x)
    sum_exp = tl.sum(exp_x, axis=0)
    attn = exp_x / sum_exp
    tl.store(attn_ptr, attn)


# Triton kernel: compute out_row[h, :] = attn @ Kc[:, :]
@triton.jit
def compute_out_row_kernel(
    attn_ptr, Kc_ptr, out_ptr, KV: tl.int32, Dn: tl.constexpr, BLOCK_J: tl.constexpr
):
    attn = tl.load(attn_ptr)  # [KV]
    out_vec = tl.zeros([Dn], dtype=tl.float32)
    for j in range(0, Dn, BLOCK_J):
        offs_j = j + tl.arange(0, BLOCK_J)
        mask_j = offs_j < Dn
        acc = tl.zeros([BLOCK_J], dtype=tl.float32)
        # out_vec[j:j+BLOCK_J] = sum_k attn[k] * Kc[k, j:j+BLOCK_J]
        for k in range(0, KV):
            kc_k = tl.load(Kc_ptr + k * Dn + offs_j, mask=mask_j, other=0.0)
            acc += attn[k] * kc_k
        out_vec += tl.where(mask_j, acc, 0.0)
    tl.store(out_ptr, out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants
        self.num_qo_heads = 16
        self.head_dim_ckv = 512
        self.head_dim_kpe = 64
        self.block_j = 128
        self.block_k = 128
        self.inv_ln2 = 1.0 / math.log(2.0)

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA."

        device = q_nope.device
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        assert num_qo_heads == self.num_qo_heads
        assert head_dim_ckv == self.head_dim_ckv
        assert head_dim_kpe == self.head_dim_kpe

        # Squeeze cache dims
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [M, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [M, 64]

        # Output buffers (float32 for compute, bfloat16 for output)
        output = torch.empty((total_q, self.num_qo_heads, self.head_dim_ckv), dtype=torch.float32, device=device)
        lse_out = torch.empty((total_q, self.num_qo_heads), dtype=torch.float32, device=device)

        B = qo_indptr.shape[0] - 1  # number of batches
        q_nope_f32 = q_nope.to(torch.float32)  # ensure float32 for compute
        q_pe_f32 = q_pe.to(torch.float32)

        for b in range(B):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            if q_start >= q_end or kv_start >= kv_end:
                continue

            # Gather Kc and Kp for this batch
            tok_idx = kv_indices[kv_start:kv_end].to(torch.int64)  # indices into cache
            Kc_batch = Kc_all[tok_idx]  # [KV, 512]
            Kp_batch = Kp_all[tok_idx]  # [KV, 64]
            KV = Kc_batch.shape[0]

            # For each query in this batch
            q_len = q_end - q_start
            for i in range(q_len):
                q_row = q_start + i  # absolute query row index in q_nope

                # Allocate intermediates
                logits = torch.empty((KV,), dtype=torch.float32, device=device)

                # Launch compute_logits_kernel for each head h
                for h in range(self.num_qo_heads):
                    logits_ptr = logits  # pass pointer to 1D logits
                    # Launch kernel
                    # Note: Triton expects int32 pointers and int32 scalars
                    compute_logits_kernel[(1,)](
                        q_nope_f32, q_pe_f32, Kc_batch, Kp_batch,
                        qo_indptr, kv_indptr, kv_indices,
                        sm_scale,
                        b, i, h,
                        total_q,
                        KV,
                        self.head_dim_ckv, self.head_dim_kpe,
                        self.block_j, self.block_k
                    )
                    # Now lse and softmax on logits
                    lse_vec = torch.empty((), dtype=torch.float32, device=device)
                    softmax_vec = torch.empty((KV,), dtype=torch.float32, device=device)
                    lse_row_kernel[(1,)](
                        logits, lse_vec, KV, self.inv_ln2
                    )
                    softmax_row_kernel[(1,)](
                        logits, softmax_vec, KV
                    )
                    # Compute out row
                    out_row = torch.empty((self.head_dim_ckv,), dtype=torch.float32, device=device)
                    compute_out_row_kernel[(1,)](
                        softmax_vec, Kc_batch, out_row, KV, self.head_dim_ckv, self.block_j
                    )
                    # Store outputs
                    output[q_row, h, :] = out_row
                    lse_out[q_row, h] = lse_vec[0]

        # Return bfloat16 output and float32 lse
        return output.to(torch.bfloat16), lse_out


def run(*args):
    return ModelNew()(*args)
