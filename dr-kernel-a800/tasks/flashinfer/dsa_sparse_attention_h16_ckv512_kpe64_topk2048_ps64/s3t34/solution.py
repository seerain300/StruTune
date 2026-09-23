import math
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_row(A_ptr, B_ptr, C_ptr,
               N, M: tl.constexpr, BLOCK_M: tl.constexpr):
    """
    Compute C[j] = sum_i A[i] * B[j, i] for j in [0, N).
    A is 1D vector of length M. B is a contiguous [N, M] matrix.
    """
    j_vec = tl.arange(0, BLOCK_M)
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    for m0 in range(0, M, BLOCK_M):
        j = m0 + j_vec
        mask = j < M
        a_chunk = tl.load(A_ptr + j, mask=mask, other=0.0)
        b_block = tl.load(B_ptr + j[:, None] * M + (m0 + j_vec)[None, :],
                          mask=mask[:, None], other=0.0)
        for ii in range(BLOCK_M):
            i = m0 + ii
            ai = a_chunk[ii]
            row = b_block[ii, :]
            acc += ai * row
    store_j = tl.arange(0, N)
    tl.store(C_ptr + store_j, acc, mask=store_j < N)


@triton.jit
def softmax_logsumexp2_row(X_ptr, Valid_ptr, Out_ptr, LSE_ptr,
                            N: tl.constexpr, TOPK: tl.constexpr):
    """
    Compute stable softmax over TOPK entries in X with validity mask Valid (int32 0/1),
    write softmax probabilities to Out_ptr (only valid entries get probability),
    and write base-2 logsumexp to LSE_ptr[0].
    """
    # Pass 1: find max over valid
    m = -float("inf")
    for j in range(0, TOPK):
        valid_j = tl.load(Valid_ptr + j)
        xj = tl.load(X_ptr + j)
        if valid_j != 0:
            m = tl.maximum(m, xj)

    # Pass 2: sum of exp(xj - m) over valid
    s = 0.0
    for j in range(0, TOPK):
        valid_j = tl.load(Valid_ptr + j)
        xj = tl.load(X_ptr + j)
        if valid_j != 0:
            s += tl.exp(xj - m)

    # LSE = m + log(s) / ln(2)
    lse_val = m + tl.log(s) * 1.4426950408889634  # 1/ln(2)
    tl.store(LSE_ptr + 0, lse_val)

    # Pass 3: write normalized softmax to Out_ptr for valid entries
    inv_s = 1.0 / s
    for j in range(0, TOPK):
        valid_j = tl.load(Valid_ptr + j)
        xj = tl.load(X_ptr + j)
        if valid_j != 0:
            out_j = tl.exp(xj - m) * inv_s
        else:
            out_j = 0.0
        tl.store(Out_ptr + j, out_j)


@triton.jit
def reduction_row(Attn_ptr, B_ptr, Out_ptr,
                  N, OUT: tl.constexpr, BLOCK_OUT: tl.constexpr, BLOCK_TOPK: tl.constexpr):
    """
    Compute Out[k] = sum_{i=0..N-1} Attn[i] * B[k, i] for k in [0, OUT).
    Attn is [N], B is [N, OUT] contiguous. We iterate over N in chunks and accumulate.
    """
    k_vec = tl.arange(0, BLOCK_OUT)
    acc = tl.zeros([BLOCK_OUT], dtype=tl.float32)
    for k0 in range(0, OUT, BLOCK_OUT):
        ks = k0 + k_vec
        mask_k = ks < OUT
        for t0 in range(0, N, BLOCK_TOPK):
            ts = t0 + tl.arange(0, BLOCK_TOPK)
            mask_t = ts < N
            attn_chunk = tl.load(Attn_ptr + ts, mask=mask_t, other=0.0)
            b_block = tl.load(B_ptr + ks[:, None] * N + ts[None, :],
                              mask=mask_k[:, None] & mask_t[None, :], other=0.0)
            acc += tl.sum(b_block * attn_chunk[None, :], axis=1)
    tl.store(Out_ptr + k_vec, acc, mask=mask_k)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        """
        Triton-only implementation:
        - q_nope: [num_tokens, num_qo_heads, 512] (bf16)
        - q_pe: [num_tokens, num_qo_heads, 64] (bf16)
        - ckv_cache: [num_pages, 64, 512] (bf16)
        - kpe_cache: [num_pages, 64, 64] (bf16)
        - sparse_indices: [num_tokens, 2048] (int32)
        - sm_scale: float
        Returns:
        - output: [num_tokens, num_qo_heads, 512] (bf16)
        - lse: [num_tokens, num_qo_heads] (float32)
        """
        device = q_nope.device

        # Shapes
        num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
        _, _, head_dim_kpe = q_pe.shape
        num_pages, page_size, _ = ckv_cache.shape
        topk = sparse_indices.shape[-1]
        assert topk == 2048, "sparse_indices' last dim must be 2048"

        # Flatten paged KV caches to [num_valid, dim]
        Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)  # [num_pages*64, 512]
        Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)  # [num_pages*64, 64]

        # Output allocation
        output = torch.empty((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((num_tokens, num_qo_heads), dtype=torch.float32, device=device)

        for t in range(num_tokens):
            indices = sparse_indices[t]  # [2048]
            valid_mask = indices != -1
            valid_indices = indices[valid_mask].to(torch.long)  # [num_valid]
            num_valid = valid_indices.numel()

            if num_valid == 0:
                # Initialize output as zeros and skip lse
                output[t] = torch.zeros((num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
                continue

            # Gather Kc_rows and Kp_rows
            Kc_rows = Kc_all[valid_indices]  # [num_valid, 512]
            Kp_rows = Kp_all[valid_indices]  # [num_valid, 64]

            # Prepare qn and qp per head, in float32 for kernels
            for h in range(num_qo_heads):
                qn = q_nope[t].to(torch.float32)  # [16, 512]
                qp = q_pe[t].to(torch.float32)    # [16, 64]
                # 1) Compute logits_qn = qn[h] @ Kc_rows.T → [num_valid]
                logits_qn = torch.empty(num_valid, dtype=torch.float32, device=device)
                matmul_row[(1,)](
                    qn[h], Kc_rows, logits_qn,
                    num_valid, 512, 256  # BLOCK_M=256
                )
                # 2) Compute logits_qp = qp[h] @ Kp_rows.T → [num_valid]
                logits_qp = torch.empty(num_valid, dtype=torch.float32, device=device)
                matmul_row[(1,)](
                    qp[h], Kp_rows, logits_qp,
                    num_valid, 64, 64  # BLOCK_M=64
                )

                # 3) Concatenate and apply mask
                X = torch.empty(2048, dtype=torch.float32, device=device)
                Valid = torch.empty(2048, dtype=torch.int32, device=device)
                X[:num_valid] = logits_qn + logits_qp  # combine contributions
                Valid[:num_valid] = 1
                if num_valid < 2048:
                    X[num_valid:] = -float("inf")
                    Valid[num_valid:] = 0

                # 4) Softmax and base-2 logsumexp
                Out = torch.empty(2048, dtype=torch.float32, device=device)
                # Triton kernel writes LSE to lse[t, h]
                softmax_logsumexp2_row[(1,)](
                    X, Valid, Out, lse[t],
                    N=num_valid, TOPK=2048
                )

                # 5) Output[h] = Attn @ Kc_rows → [512]
                attn = Out  # softmax probabilities for valid entries
                Out_final = torch.empty(head_dim_ckv, dtype=torch.float32, device=device)
                reduction_row[(1,)](
                    attn[:num_valid], Kc_rows, Out_final,
                    N=num_valid, OUT=head_dim_ckv, BLOCK_OUT=256, BLOCK_TOPK=256
                )
                output[t, h] = Out_final.to(torch.bfloat16)

        return output, lse

# Example input helpers for testing
def get_inputs():
    # Ensure tensors are on CUDA for Triton
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([8462, 64, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([8462, 64, 64], dtype=torch.bfloat16, device='cuda')
    sparse_indices = torch.randint(0, 541568, [1, 2048], dtype=torch.int32, device='cuda')
    sm_scale = 1.0
    return [q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale]

def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
