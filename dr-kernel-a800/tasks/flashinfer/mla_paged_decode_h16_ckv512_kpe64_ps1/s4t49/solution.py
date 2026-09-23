import torch
import triton
import triton.language as tl


@triton.jit
def matvec_row_kernel(
    A_ptr,       # *float32, pointer to q vector (length K), flattened to 1D
    B_ptr,       # *float32, pointer to K rows, shape [M, K], contiguous
    C_ptr,       # *float32, pointer to output, shape [M], contiguous
    M,           # int, number of rows in B (runtime)
    K: tl.constexpr,           # int, length of q and columns of B (compile-time)
    BLOCK_K: tl.constexpr = 64 # tile size along K
):
    # One program computes a single output element: C[i] = A @ B[i, :]
    i = tl.program_id(0)  # index over M
    # Guard: if i >= M, return (masking handled on host; but keep safe here)
    acc = 0.0
    # Loop over K in tiles (compile-time loops)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)  # constexpr length vector
        mask_k = offs_k < K
        a = tl.load(A_ptr + offs_k, mask=mask_k, other=0.0)       # q chunk
        b = tl.load(B_ptr + i * K + offs_k, mask=mask_k, other=0.0)  # B row chunk
        acc += tl.sum(a * b, axis=0)
    # Store the result for this i
    tl.store(C_ptr + i, acc)


@triton.jit
def softmax_lse_kernel(
    logits_ptr,          # *float32, pointer to logits per token, shape [M]
    L,                   # int, effective length (runtime)
    out_attn_ptr,        # *float32, pointer to output attn per token, shape [M]
    out_lse_ptr,         # *float32, pointer to output lse per head, shape [1]
    BLOCK_M: tl.constexpr = 128,
    M_CONST: tl.constexpr = 1024,  # upper bound for M (compile-time)
    SM_SCALE: tl.constexpr = 1,    # scaling factor for logits
):
    # First pass: find max of scaled logits over first L elements (ignore out-of-range via mask)
    max_val = -float("inf")
    for m0 in range(0, M_CONST, BLOCK_M):
        offs = m0 + tl.arange(0, BLOCK_M)
        mask = offs < L
        x = tl.load(logits_ptr + offs, mask=mask, other=-float("inf"))
        x = x * SM_SCALE
        local_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, local_max)

    # Second pass: compute sum of exp(scaled_logits - max_val) over L and write attn
    sum_exp = 0.0
    for m0 in range(0, M_CONST, BLOCK_M):
        offs = m0 + tl.arange(0, BLOCK_M)
        mask = offs < L
        x = tl.load(logits_ptr + offs, mask=mask, other=0.0) * SM_SCALE
        e = tl.exp(x - max_val)
        sum_exp += tl.sum(e, axis=0)
        # Write attn for these tokens: e / sum_exp
        attn_chunk = e / sum_exp
        tl.store(out_attn_ptr + offs, attn_chunk, mask=mask)

    lse = max_val + tl.log(sum_exp) / 0.6931471805599453  # log(2)
    tl.store(out_lse_ptr, lse)


@triton.jit
def matvec_out_kernel(
    attn_ptr,            # *float32, pointer to attn per token, shape [M]
    B_ptr,               # *float32, pointer to K rows, shape [M, Kc_DIM], contiguous
    C_ptr,               # *float32, pointer to output, shape [Kc_DIM], contiguous
    M,                   # int, number of rows in B (runtime)
    Kc_DIM: tl.constexpr,            # int, output length (compile-time)
    BLOCK_M: tl.constexpr = 128,
    M_CONST: tl.constexpr = 1024     # upper bound for M (compile-time)
):
    # One program computes one output element: C[j] = sum_i attn[i] * B[i, j]
    j = tl.program_id(0)  # index over Kc_DIM
    acc = 0.0
    # Loop over M in tiles
    for m0 in range(0, M_CONST, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)  # constexpr vector
        mask_m = offs_m < M
        attn_chunk = tl.load(attn_ptr + offs_m, mask=mask_m, other=0.0)
        b_chunk = tl.load(B_ptr + offs_m * Kc_DIM + j, mask=mask_m, other=0.0)
        acc += tl.sum(attn_chunk * b_chunk, axis=0)
    tl.store(C_ptr + j, acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes
        batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages = ckv_cache.shape[0]
        # Constants
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64
        assert ckv_cache.shape == kpe_cache.shape  # [num_pages, 1, dim]
        device = q_nope.device
        # Prepare Kc_all and Kp_all: [num_pages, dim]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_ckv]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_kpe]

        # Output buffers
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # sm_scale must be float32 scalar
        sm_scale_val = float(sm_scale)

        for b in range(batch_size):
            # If no valid tokens, skip
            if kv_indptr.shape[0] <= b + 1 or kv_indptr[b + 1].item() <= kv_indptr[b].item():
                lse[b] = -float("inf")
                # Output zeros
                output[b] = 0.0
                continue

            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            M = max(0, page_end - page_beg)
            if M == 0:
                lse[b] = -float("inf")
                output[b] = 0.0
                continue

            # Gather tokens and corresponding rows
            tok_idx = kv_indices[page_beg:page_end]  # [M]
            Kc = Kc_all[tok_idx].contiguous()        # [M, 512]
            Kp = Kp_all[tok_idx].contiguous()        # [M, 64]

            # q vectors
            qn = q_nope[b].to(torch.float32).contiguous()  # [512]
            qp = q_pe[b].to(torch.float32).contiguous()    # [64]

            # Compute logits for each token i: [M]
            logits = torch.empty((M,), dtype=torch.float32, device=device)
            # Launch matvec_row_kernel: grid=(M,)
            grid_log = (M,)
            matvec_row_kernel[grid_log](
                qn, Kc, logits, M, head_dim_ckv, 64
            )

            # Add contribution from Kp
            logits_p = torch.empty((M,), dtype=torch.float32, device=device)
            grid_log_p = (M,)
            matvec_row_kernel[grid_log_p](
                qp, Kp, logits_p, M, head_dim_kpe, 64
            )
            logits = logits + logits_p

            # Scale logits
            logits_scaled = logits * sm_scale_val

            # Compute softmax and lse per head
            attn = torch.empty((M,), dtype=torch.float32, device=device)
            lse_val = torch.empty((1,), dtype=torch.float32, device=device)
            grid_soft = (1,)
            softmax_lse_kernel[grid_soft](
                logits_scaled, M, attn, lse_val, 128, 1024, sm_scale_val
            )
            lse[b] = lse_val[0]

            # Compute out = attn @ Kc, shape [512]
            out_vec = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
            grid_out = (head_dim_ckv,)
            matvec_out_kernel[grid_out](
                attn, Kc, out_vec, M, head_dim_ckv, 128, 1024
            )
            output[b] = out_vec

        # Cast output to bfloat16 to match original
        output = output.to(torch.bfloat16)
        return output, lse


# Optional helpers (not used by evaluator, but shown for completeness)
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
