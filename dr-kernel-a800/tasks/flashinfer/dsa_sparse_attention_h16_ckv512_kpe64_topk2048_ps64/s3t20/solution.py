import math
import torch
import triton
import triton.language as tl


# Triton kernel: row-wise matmul C = A_row @ B, where:
# A_row is 1D vector of length M.
# B is 2D matrix of shape (N, M). Output C is 1D vector of length N.
# We use TOPK as a compile-time constant and mask j < N to avoid invalid loads.
@triton.jit
def matmul_row(A_ptr, B_ptr, C_ptr,
               M: tl.constexpr, N: tl.constexpr, TOPK: tl.constexpr):
    # Vector j over TOPK
    j = tl.arange(0, TOPK)
    mask_j = j < N  # only first N columns are valid
    # Load A row (length M), contiguous
    a = tl.load(A_ptr)
    # Load B block: shape [TOPK, M]
    b_ptrs = B_ptr + j[:, None] * M + tl.arange(0, M)
    b = tl.load(b_ptrs, mask=mask_j[:, None], other=0.0)
    # Accumulate dot products over M
    acc = tl.zeros([TOPK], dtype=tl.float32)
    for m in range(0, M):
        acc += b[:, m] * a[m]
    # Store results only for valid j
    tl.store(C_ptr + j, acc, mask=mask_j)


# Triton kernel: row-wise softmax (numerically stable) and base-2 logsumexp.
# X is 1D vector of length TOPK (e.g., 2048).
# Valid is 1D mask (int32) of length TOPK: 1 for valid, 0 for padded.
# Outputs:
#   Out: softmax probabilities (float32) of length TOPK (valid positions are meaningful).
#   LSE_ptr: scalar (float32) storing logsumexp base-2 for this row.
@triton.jit
def softmax_logsumexp2_row(X_ptr, Valid_ptr, Out_ptr, LSE_ptr,
                            TOPK: tl.constexpr):
    # Compute max over valid entries using vectorized ops
    valid_vec = tl.load(Valid_ptr)  # int32 [TOPK]
    x = tl.load(X_ptr)              # float32 [TOPK]
    x = tl.where(valid_vec != 0, x, -float("inf"))
    m = tl.max(x, axis=0)
    # Compute sum of exp over valid entries
    s = tl.sum(tl.exp(x - m), axis=0)
    # Write softmax probabilities for valid entries
    prob = tl.exp(x - m) / s
    valid_mask = valid_vec != 0
    tl.store(Out_ptr, prob, mask=valid_mask)
    # Base-2 logsumexp: lse = m + log(s); divide by ln(2)
    lse_val = m + tl.log(s)
    tl.store(LSE_ptr, lse_val / 1.4426950408889634)  # 1 / ln(2)


# Triton kernel: compute out = Attn @ Kc_row
# Attn is a 1D vector of length NUM_VALID (constexpr, e.g., 2048).
# Kc is a 2D matrix of shape (NUM_VALID, OUT), OUT is tl.constexpr (e.g., 512).
# out is a 1D vector of length OUT, computed as sum_i Attn[i] * Kc[i, :].
@triton.jit
def reduction_row(Attn_ptr, Kc_ptr, Out_ptr,
                  NUM_VALID: tl.constexpr, OUT: tl.constexpr):
    # Initialize output accumulator
    acc = tl.zeros([OUT], dtype=tl.float32)
    # Loop over columns and accumulate
    for o in range(0, OUT):
        sum_val = 0.0
        # Load Kc column o for all j in [0, NUM_VALID)
        j = tl.arange(0, NUM_VALID)
        kc_col = tl.load(Kc_ptr + j * OUT + o)
        # Multiply and reduce
        for jj in range(0, NUM_VALID):
            sum_val += tl.load(Attn_ptr + jj) * kc_col[jj]
        acc[o] = sum_val
    tl.store(Out_ptr, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_qo_heads = 16
        self.head_dim_ckv = 512
        self.head_dim_kpe = 64
        self.topk = 2048  # maximum number of candidates per token

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # Validate shapes
        num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
        assert num_qo_heads == self.num_qo_heads
        assert head_dim_ckv == self.head_dim_ckv
        assert q_pe.shape[1] == self.num_qo_heads
        assert q_pe.shape[-1] == self.head_dim_kpe
        assert ckv_cache.shape[1] == 64 and ckv_cache.shape[-1] == self.head_dim_ckv
        assert kpe_cache.shape[1] == 64 and kpe_cache.shape[-1] == self.head_dim_kpe

        # Ensure CUDA tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA."
        device = q_nope.device

        # Flatten KV caches to token-level candidates
        Kc_all = ckv_cache.reshape(-1, self.head_dim_ckv).to(torch.float32)  # [num_pages*64, 512]
        Kp_all = kpe_cache.reshape(-1, self.head_dim_kpe).to(torch.float32)  # [num_pages*64, 64]

        # Output buffers
        output = torch.empty((num_tokens, self.num_qo_heads, self.head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((num_tokens, self.num_qo_heads), dtype=torch.float32, device=device)

        for t in range(num_tokens):
            indices = sparse_indices[t]  # [topk]
            valid_mask = indices != -1
            valid_indices = indices[valid_mask].to(torch.long)
            num_valid = valid_indices.numel()

            # Validity mask for Triton softmax (int32), length TOPK
            valid_int = valid_mask.to(torch.int32)  # device tensor

            # Gather Kc and Kp rows for this token
            Kc = Kc_all[valid_indices]  # [num_valid, 512]
            Kp = Kp_all[valid_indices]  # [num_valid, 64]

            # q_nope and q_pe for this token (dtype float32 for compute)
            qn = q_nope[t].to(torch.float32)  # [num_qo_heads, 512]
            qp = q_pe[t].to(torch.float32)    # [num_qo_heads, 64]

            for h in range(self.num_qo_heads):
                # 1) Compute logits[h] = (qn[h] @ Kc.T) + (qp[h] @ Kp.T)
                logits_qn = torch.empty((self.topk,), dtype=torch.float32, device=device)
                matmul_row[(1,)](qn[h], Kc, logits_qn, M=self.head_dim_ckv, N=num_valid, TOPK=self.topk)

                logits_qp = torch.empty((self.topk,), dtype=torch.float32, device=device)
                matmul_row[(1,)](qp[h], Kp, logits_qp, M=self.head_dim_kpe, N=num_valid, TOPK=self.topk)

                logits = logits_qn + logits_qp  # [topk], only first num_valid entries are meaningful
                logits_scaled = logits * sm_scale

                # 2) Softmax (base-2 logsumexp) on logits_scaled
                attn = torch.empty((self.topk,), dtype=torch.float32, device=device)
                lse_scalar = torch.empty((), dtype=torch.float32, device=device)
                softmax_logsumexp2_row[(1,)](logits_scaled, valid_int, attn, lse_scalar, TOPK=self.topk)

                # 3) Output = attn @ Kc for this head
                out_vec = torch.empty((self.head_dim_ckv,), dtype=torch.float32, device=device)
                reduction_row[(1,)](attn, Kc, out_vec, NUM_VALID=num_valid, OUT=self.head_dim_ckv)

                output[t, h] = out_vec
                lse[t, h] = lse_scalar

        # Cast output to bfloat16 to match original
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
