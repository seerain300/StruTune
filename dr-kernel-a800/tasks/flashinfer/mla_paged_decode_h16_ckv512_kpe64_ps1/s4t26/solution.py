import math
import torch
import triton
import triton.language as tl


@triton.jit
def matvec_row_kernel(
    A_ptr,      # *float32, pointer to [1, K] row vector (we pass a 1xK pointer)
    B_ptr,      # *float32, pointer to [M_CONST, K] matrix
    C_ptr,      # *float32, pointer to [1, M_CONST] output row
    K: tl.constexpr,         # int, reduction dimension (e.g., 512 or 64)
    BLOCK_K: tl.constexpr,   # tile size along K (e.g., 128 or 64)
):
    # Each program computes one output element C[0, i] for i in [0, M_CONST)
    i = tl.program_id(0)
    acc = 0.0
    # Loop over K in tiles using static_range
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)  # vector of size BLOCK_K
        mask_k = k < K
        # Load A chunk: A is a 1xK row; load the single row vector
        a = tl.load(A_ptr + k, mask=mask_k, other=0.0)  # [BLOCK_K]
        # Load B chunk for row i: B[i, k]
        b = tl.load(B_ptr + i * K + k, mask=mask_k, other=0.0)  # [BLOCK_K]
        # Accumulate dot product
        acc += tl.sum(a * b, axis=0)
    # Store result to C[0, i]
    tl.store(C_ptr + i, acc)


@triton.jit
def matvec_attn_kernel(
    A_ptr,      # *float32, pointer to [M_CONST] attention vector
    B_ptr,      # *float32, pointer to [M_CONST, Kc_dim] matrix (Kc rows)
    C_ptr,      # *float32, pointer to [1, Kc_dim] output row
    M_CONST: tl.constexpr,     # int, number of rows in A (compile-time for loop)
    Kc_dim: tl.constexpr,      # int, number of columns in B (compile-time for loop)
    BLOCK_M: tl.constexpr,     # tile size along M, e.g., 128
):
    # Each program computes one output element C[0, d] for d in [0, Kc_dim)
    d = tl.program_id(0)
    acc = 0.0
    # Loop over M in tiles and accumulate A[m] * B[m, d]
    for m0 in range(0, M_CONST, BLOCK_M):
        m = m0 + tl.arange(0, BLOCK_M)  # vector of size BLOCK_M
        mask_m = m < M_CONST
        a = tl.load(A_ptr + m, mask=mask_m, other=0.0)  # [BLOCK_M]
        b = tl.load(B_ptr + m * Kc_dim + d, mask=mask_m, other=0.0)  # [BLOCK_M]
        acc += tl.sum(a * b, axis=0)
    tl.store(C_ptr + d, acc)


@triton.jit
def softmax_lse_kernel(
    x_ptr,           # *float32, pointer to logits_scaled vector [M_CONST]
    out_lse_ptr,     # *float32, pointer to lse output [1] (per head)
    attn_ptr,        # *float32, pointer to attn vector [M_CONST]
    M_CONST: tl.constexpr,    # int, length of x
    sm_scale,        # float32
    BLOCK_M: tl.constexpr,    # int, block size for vectorized reduction
):
    # We'll perform vectorized reductions over chunks of size BLOCK_M.
    # Initialize vector accumulators
    max_vec = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    sum_exp = 0.0

    # First pass: compute max over chunks
    for m0 in range(0, M_CONST, BLOCK_M):
        m = m0 + tl.arange(0, BLOCK_M)
        mask = m < M_CONST
        x = tl.load(x_ptr + m, mask=mask, other=-float("inf"))
        max_vec = tl.maximum(max_vec, x)
    max_val = tl.max(max_vec, axis=0)

    # Second pass: compute sum of exp(sm_scale * x - max_val) over chunks
    for m0 in range(0, M_CONST, BLOCK_M):
        m = m0 + tl.arange(0, BLOCK_M)
        mask = m < M_CONST
        x = tl.load(x_ptr + m, mask=mask, other=-float("inf"))
        e = tl.exp((x - max_val) * sm_scale)
        sum_exp += tl.sum(e, axis=0)

    lse_val = tl.log(sum_exp) / math.log(2.0)  # convert ln to log2
    tl.store(out_lse_ptr, lse_val)

    # Third pass: write attn vector
    for m0 in range(0, M_CONST, BLOCK_M):
        m = m0 + tl.arange(0, BLOCK_M)
        mask = m < M_CONST
        x = tl.load(x_ptr + m, mask=mask, other=-float("inf"))
        attn_chunk = tl.exp((x - max_val) * sm_scale) / sum_exp
        tl.store(attn_ptr + m, attn_chunk, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on the same device
        device = q_nope.device
        # We'll compute in float32 and cast output to bfloat16
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

        B = q_nope.shape[0]
        H = q_nope.shape[1]
        assert H == 16, "num_qo_heads must be 16"
        Kc_dim = q_nope.shape[2]
        assert Kc_dim == 512, "head_dim_ckv must be 512"
        Kp_dim = q_pe.shape[2]
        assert Kp_dim == 64, "head_dim_kpe must be 64"

        # Output buffers
        output = torch.empty((B, H, Kc_dim), dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Process each batch element
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            M = end - start
            if M <= 0:
                output[b].zero_()
                lse[b] = -float("inf")
                continue

            # Gather Kc and Kp rows for this batch
            tok_idx = kv_indices[start:end].to(torch.int32)
            Kc = Kc_all[tok_idx]  # [M, 512]
            Kp = Kp_all[tok_idx]  # [M, 64]

            # Prepare per-head vectors
            for h in range(H):
                qn = q_nope[b, h, :].to(torch.float32)  # [512]
                qp = q_pe[b, h, :].to(torch.float32)    # [64]

                # 1) Compute logits = qn @ Kc.T (M_CONST vector)
                logits = torch.empty((M,), dtype=torch.float32, device=device)
                grid = (M,)  # M_CONST is runtime; Triton will handle masks
                matvec_row_kernel[grid](
                    qn, Kc, logits,
                    K=Kc_dim, BLOCK_K=128,
                    num_warps=4, num_stages=2
                )
                # 2) Compute logits_qp = qp @ Kp.T (M_CONST vector)
                logits_qp = torch.empty((M,), dtype=torch.float32, device=device)
                matvec_row_kernel[grid](
                    qp, Kp, logits_qp,
                    K=Kp_dim, BLOCK_K=64,
                    num_warps=4, num_stages=2
                )
                logits = logits + logits_qp  # [M]

                # 3) lse and attn via Triton softmax_lse_kernel (per head)
                # Make logits_scaled contiguous
                logits_scaled = logits  # already computed scaled in kernel by sm_scale? Wait:
                # We need to multiply by sm_scale. We'll pass logits and apply in kernel via sm_scale.
                # Create x_ptr for Triton; we can reuse logits buffer and multiply by sm_scale inside kernel (we passed sm_scale).
                # However, Triton expects raw pointer; we can create a new tensor for x_ptr:
                x = logits  # [M]
                attn = torch.empty((M,), dtype=torch.float32, device=device)
                # Launch softmax_lse_kernel with BLOCK_M >= M; choose 2048 to cover typical sizes
                softmax_lse_kernel[(1,)](
                    x, lse[b].unsqueeze(0), attn,
                    M_CONST=M, sm_scale=sm_scale, BLOCK_M=2048,
                    num_warps=4, num_stages=2
                )
                # 4) out[h, :] = attn @ Kc (matvec across M)
                out_vec = torch.empty((Kc_dim,), dtype=torch.float32, device=device)
                grid_out = (Kc_dim,)
                matvec_attn_kernel[grid_out](
                    attn, Kc, out_vec,
                    M_CONST=M, Kc_dim=Kc_dim, BLOCK_M=128,
                    num_warps=4, num_stages=2
                )
                output[b, h, :] = out_vec

        # Cast output to bfloat16 as in original
        return output.to(torch.bfloat16), lse

# Optional: helpers (not used by evaluator)
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

def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
