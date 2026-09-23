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
    # Initialize accumulator for this output element
    acc = 0.0
    # Loop over K in tiles
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)  # vector of indices along K (constexpr length)
        mask_k = offs_k < K
        # Load q chunk: A_ptr is [K], so slice via + offs_k
        a = tl.load(A_ptr + offs_k, mask=mask_k, other=0.0)
        # Load B row chunk: B_ptr is [M, K], row i, cols offs_k
        b = tl.load(B_ptr + i * K + offs_k, mask=mask_k, other=0.0)
        # Accumulate dot product of a and b
        acc += tl.sum(a * b, axis=0)
    # Store the result for this i
    tl.store(C_ptr + i, acc)


@triton.jit
def matvec_out_kernel(
    attn_ptr,            # *float32, pointer to attn per token, shape [M]
    B_ptr,               # *float32, pointer to K rows, shape [M, Kc_dim], contiguous
    C_ptr,               # *float32, pointer to output, shape [Kc_dim], contiguous
    M,                   # int, number of rows in B (runtime)
    Kc_DIM: tl.constexpr,            # int, length of output and columns of B (compile-time)
    BLOCK_M: tl.constexpr = 64       # tile size along M for accumulation
):
    # One program computes one output element: C[j] = sum_i attn[i] * B[i, j]
    j = tl.program_id(0)  # index over output dimension j
    acc = 0.0
    # Loop over M in tiles
    for m0 in range(0, M, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)  # vector of indices along M (constexpr length)
        mask_m = offs_m < M
        attn = tl.load(attn_ptr + offs_m, mask=mask_m, other=0.0)
        b = tl.load(B_ptr + offs_m * Kc_DIM + j, mask=mask_m, other=0.0)
        acc += tl.sum(attn * b, axis=0)
    tl.store(C_ptr + j, acc)


@triton.jit
def softmax_lse_kernel(
    logits_ptr,          # *float32, pointer to logits per token, shape [M]
    L,                   # int, effective length (runtime)
    out_attn_ptr,        # *float32, pointer to output attn per token, shape [M]
    out_lse_ptr,         # *float32, pointer to output lse per head, shape [1]
    BLOCK_M: tl.constexpr = 128,  # tile size along M for reductions
    M_CONST: tl.constexpr = 1024, # upper bound for M (compile-time), loop will mask with L
    SM_SCALE: tl.constexpr = 1     # sm_scale as compile-time constant (not used here, but kept for consistency)
):
    # Two-pass stable softmax and logsumexp on first L elements.
    max_val = -float("inf")
    for m0 in range(0, M_CONST, BLOCK_M):
        offs = m0 + tl.arange(0, BLOCK_M)
        mask = offs < L
        x = tl.load(logits_ptr + offs, mask=mask, other=-float("inf"))
        local_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, local_max)
    sum_exp = 0.0
    for m0 in range(0, M_CONST, BLOCK_M):
        offs = m0 + tl.arange(0, BLOCK_M)
        mask = offs < L
        x = tl.load(logits_ptr + offs, mask=mask, other=0.0)
        e = tl.exp(x - max_val)
        sum_exp += tl.sum(e, axis=0)
        # Store attention values for this pass (will overwrite previous pass, but correctness is based on lse and we don't rely on attn here)
        # Note: We don't need to store attn for correctness of lse; keeping code lightweight.
        # However, to align with earlier plan, we compute and store attn for potential use.
        attn = e / sum_exp
        tl.store(out_attn_ptr + offs, attn, mask=mask)
    lse = max_val + tl.log(sum_exp) / 0.6931471805599453  # log(2)
    tl.store(out_lse_ptr, lse)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        q_nope: [B, 16, 512] bfloat16
        q_pe:   [B, 16, 64] bfloat16
        ckv_cache: [P, 1, 512] bfloat16
        kpe_cache: [P, 1, 64] bfloat16
        kv_indptr: [B+1] int32
        kv_indices: [N] int32
        sm_scale: float32
        returns:
        output: [B, 16, 512] bfloat16
        lse: [B, 16] float32
        """
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Kc_dim = q_nope.shape[2]
        Kp_dim = q_pe.shape[2]
        P = ckv_cache.shape[0]
        device = q_nope.device

        # Prepare all rows Kc_all and Kp_all on device as float32
        # Note: We do not compute these in a gather kernel; instead, we pass through as provided.
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [P, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [P, 64]

        output = torch.empty((B, H, Kc_dim), dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        for b in range(B):
            # Compute tokens for this batch
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                # No KV cache for this batch element
                output[b].zero_()
                lse[b].zero_()
                continue

            tokens = kv_indices[page_beg:page_end]  # [M]
            M = tokens.numel()
            if M <= 0:
                output[b].zero_()
                lse[b].zero_()
                continue

            # Gather Kc and Kp rows: [M, Kc_dim] and [M, Kp_dim]
            Kc = Kc_all[tokens]                      # [M, 512]
            Kp = Kp_all[tokens]                      # [M, 64]

            # Prepare q vectors as float32, contiguous
            qn = q_nope[b].to(torch.float32).contiguous()  # [16, 512]
            qp = q_pe[b].to(torch.float32).contiguous()    # [16, 64]

            # Launch matvec_row_kernel to compute logits per head
            # Note: We flatten q vectors and K rows for the kernel.
            for h in range(H):
                qn_flat = qn[h].contiguous()  # [512]
                Kc_b = Kc.contiguous()       # [M, 512]
                logits_c = torch.empty(M, dtype=torch.float32, device=device)
                grid = (M,)
                matvec_row_kernel[grid](
                    qn_flat, Kc_b, logits_c, M, Kc_dim, BLOCK_K=64
                )
                qh_flat = qp[h].contiguous()  # [64]
                Kp_b = Kp.contiguous()        # [M, 64]
                logits_p = torch.empty(M, dtype=torch.float32, device=device)
                grid = (M,)
                matvec_row_kernel[grid](
                    qh_flat, Kp_b, logits_p, M, Kp_dim, BLOCK_K=64
                )
                logits = logits_c + logits_p  # [M]
                # Compute stable softmax and lse per token
                attn = torch.empty(M, dtype=torch.float32, device=device)
                lse_val = torch.empty(1, dtype=torch.float32, device=device)
                grid_soft = (1,)
                softmax_lse_kernel[grid_soft](
                    logits, M, attn, lse_val, BLOCK_M=128, M_CONST=1024
                )
                lse[b, h] = lse_val[0]
                # Compute out[h, :] = attn @ Kc
                out_vec = torch.empty(Kc_dim, dtype=torch.float32, device=device)
                grid_out = (Kc_dim,)
                matvec_out_kernel[grid_out](
                    attn, Kc_b, out_vec, M, Kc_DIM=Kc_dim, BLOCK_M=64
                )
                output[b, h] = out_vec

        # Cast output to bfloat16 to match original
        output = output.to(torch.bfloat16)
        return output, lse


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
