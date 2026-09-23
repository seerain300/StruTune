import math
import torch

import triton
import triton.language as tl


# Triton kernel: compute C = A_row @ B, where
# A_ptr points to [1, K] (we pass a 1xK pointer, effectively a row vector),
# B_ptr points to [M, K], C_ptr points to [1, M].
# We launch grid=(1,) since we only compute one output row.
@triton.jit
def matvec_row_kernel(
    A_ptr,          # *float32, shape [1, K] (we pass a 1xK pointer, effectively a row vector)
    B_ptr,          # *float32, shape [M, K]
    C_ptr,          # *float32, shape [1, M]
    K: tl.constexpr,       # compile-time K (e.g., 512)
    M,               # runtime int: number of rows in B
    BLOCK_K: tl.constexpr  # tile size along K (e.g., 64 or 128)
):
    # One program computes the row-vector matmul: result has length M
    # We accumulate into a vector acc of length M.
    acc = tl.zeros((M,), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    # Loop over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + offs_k
        # Load A_row[k] as a vector of size BLOCK_K
        a_vec = tl.load(A_ptr + k_idx)  # A_ptr + k_idx gives [BLOCK_K] vector

        # Initialize a partial acc_vec of size M to accumulate inner products
        acc_vec = tl.zeros((M,), dtype=tl.float32)

        # For each row j in [0, M), compute dot with B[j, k0:k0+BLOCK_K]
        # Triton supports elementwise pointer arithmetic.
        for j in range(0, M):
            b_vec = tl.load(B_ptr + j * K + k_idx)  # [BLOCK_K]
            acc_vec[j] = tl.sum(a_vec * b_vec, axis=0)

        acc += acc_vec

    # Write result to C[0, :]
    for m in range(0, M):
        tl.store(C_ptr + m, acc[m])


# Triton kernel: compute out = attn @ Kc, i.e., row-vector matmul A=attn [1, M], B=Kc [M, K], C[out] [1, K]
# We tile along M with BLOCK_M as constexpr.
@triton.jit
def matvec_row_kernel_out(
    A_ptr,            # *float32, shape [1, M]
    B_ptr,            # *float32, shape [M, K]
    C_ptr,            # *float32, shape [1, K]
    M: tl.constexpr,  # compile-time M (we pass the exact M as meta-parameter)
    K: tl.constexpr,  # compile-time K (e.g., 512)
    BLOCK_M: tl.constexpr  # tile size along M (e.g., 128)
):
    acc = tl.zeros((K,), dtype=tl.float32)

    offs_k = tl.arange(0, K)

    for m0 in range(0, M, BLOCK_M):
        m_idx = m0 + tl.arange(0, BLOCK_M)
        mask = m_idx < M
        a_vec = tl.load(A_ptr + m_idx, mask=mask, other=0.0)  # [BLOCK_M]
        acc_vec = tl.zeros((K,), dtype=tl.float32)
        for mm in range(0, BLOCK_M):
            mi = m0 + mm
            if mi < M:
                b_vec = tl.load(B_ptr + mi * K + offs_k)  # [K]
                acc_vec += a_vec[mm] * b_vec
        acc += acc_vec

    for k in range(0, K):
        tl.store(C_ptr + k, acc[k])


class ModelNew(torch.nn.Module):
    def __init__(self, block_k: int = 128, block_m: int = 128):
        super().__init__()
        self.block_k = block_k
        self.block_m = block_m

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes
        B, H, Kc_dim = q_nope.shape
        _, _, Kp_dim = q_pe.shape
        device = q_nope.device

        # Prepare caches as [num_tokens, dim]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [N, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [N, 64]

        output = torch.zeros((B, H, Kc_dim), dtype=torch.bfloat16, device=device)
        lse = torch.full((B, H), -float("inf"), dtype=torch.float32, device=device)

        for b in range(B):
            # Determine valid token range
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            M = end - start
            if M <= 0:
                # No valid tokens for this batch element
                continue

            # Gather Kc and Kp rows for this batch
            tok_idx = kv_indices[start:end].to(torch.int64)
            Kc = Kc_all[tok_idx]  # [M, 512]
            Kp = Kp_all[tok_idx]  # [M, 64]

            # Convert q vectors to float32 and contiguous
            qn = q_nope[b].to(torch.float32).contiguous()  # [16, 512]
            qp = q_pe[b].to(torch.float32).contiguous()    # [16, 64]

            # Compute logits per head using Triton matvec kernels
            for h in range(H):
                qn_vec = qn[h].contiguous().view(1, -1)  # [1, 512]
                qp_vec = qp[h].contiguous().view(1, -1)  # [1, 64]

                # Temporary tensors to hold logits
                logits_qn = torch.empty((1, M), dtype=torch.float32, device=device)
                logits_qp = torch.empty((1, M), dtype=torch.float32, device=device)

                # Launch matvec kernel for qn @ Kc.T
                matvec_row_kernel[(1,)](
                    qn_vec, Kc, logits_qn,
                    K=Kc_dim, M=M, BLOCK_K=self.block_k
                )
                # Launch matvec kernel for qp @ Kp.T
                matvec_row_kernel[(1,)](
                    qp_vec, Kp, logits_qp,
                    K=Kp_dim, M=M, BLOCK_K=64  # Kp_dim == 64
                )

                # Sum and scale
                logits = logits_qn + logits_qp  # [1, M]
                logits_scaled = logits * sm_scale  # [1, M]

                # Compute lse and attn using torch reductions (avoid Triton tl.arange with runtime M).
                # lse[h] = logsumexp(logits_scaled) / ln(2) per head
                max_val = torch.max(logits_scaled)
                sum_exp = torch.sum(torch.exp(logits_scaled - max_val))
                lse[b, h] = torch.log(sum_exp) / math.log(2.0)

                # attn = softmax(logits_scaled, dim=-1)
                attn = torch.exp(logits_scaled - max_val) / sum_exp  # [1, M]

                # Compute out[h, :] = attn @ Kc (matvec), Triton kernel
                attn_vec = attn.view(1, M)  # [1, M]
                out_vec = torch.empty((1, Kc_dim), dtype=torch.float32, device=device)
                matvec_row_kernel_out[(1,)](
                    attn_vec, Kc, out_vec,
                    M=M, K=Kc_dim, BLOCK_M=self.block_m
                )
                output[b, h] = out_vec[0].to(torch.bfloat16)

        return output, lse


# Helper to generate inputs (same as original)
def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16)
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16)
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16)
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16)
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32)
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


# Fused function (optional)
def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


# Original Model interface
class Model(torch.nn.Module):
    def forward(self, *args):
        return ModelNew().forward(*args)


def run(*args):
    return ModelNew()(*args)
