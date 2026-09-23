import torch
import triton
import triton.language as tl


# Kernel: copy a row from a 3D tensor A[T, M, K] to a 2D fp32 buffer B[T, M*K] (row-major flattened).
# A_ptr: *const float, B_ptr: *float
# We launch one program per row (pid_t in [0, T)) and copy the entire [M, K] plane as a contiguous vector of length M*K.
@triton.jit
def copy_row_to_fp32_kernel(A_ptr, B_ptr,
                            T, M, K,
                            stride_at, stride_am, stride_ak,
                            stride_bt, stride_bm,
                            BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_t = tl.program_id(0)  # row index in T
    total = M * K
    offs = tl.arange(0, BLOCK_M) * BLOCK_K + tl.arange(0, BLOCK_K)  # linear offsets in [0, total)
    mask = offs < total
    # Base pointer for this row in A: A_row_ptr = A_ptr + pid_t * stride_at
    A_row_ptr = A_ptr + pid_t * stride_at
    # Build pointer to A for offs: unravel offs -> (m, k): m = offs // K, k = offs % K
    m = offs // K
    k = offs - m * K  # faster than modulo
    A_vec_ptr = A_row_ptr + m * stride_am + k * stride_ak
    a = tl.load(A_vec_ptr, mask=mask, other=0.0)
    # Store to B row at time index pid_t: B row is contiguous, stride_bt is row stride, stride_bm is column stride=1
    B_row_ptr = B_ptr + pid_t * stride_bt
    tl.store(B_row_ptr + offs * stride_bm, a, mask=mask)


# Kernel: left multiply A[M, N] @ B[N, K]^T -> C[M, K]
# A_ptr: [M, N], B_ptr: [N, K], C_ptr: [M, K]
@triton.jit
def matmul_left_kernel(A_ptr, B_ptr, C_ptr,
                       M, N, K,
                       stride_am, stride_an, stride_ak,
                       stride_bn, stride_bk, stride_cn, stride_cm,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    # Loop over N in tiles
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        # A_tile: [BLOCK_M, BLOCK_N]
        A_tile_ptr = A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
        A_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        A_tile = tl.load(A_tile_ptr, mask=A_mask, other=0.0)

        # B_tile: [BLOCK_N, BLOCK_K] (we want B^T where B is [N, K])
        B_tile_ptr = B_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk
        B_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        B_tile = tl.load(B_tile_ptr, mask=B_mask, other=0.0)

        # acc += A_tile @ B_tile
        acc += tl.dot(A_tile, B_tile)

    # Store acc to C
    C_tile_ptr = C_ptr + offs_m[:, None] * stride_cm + offs_k[None, :] * stride_cn
    C_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(C_tile_ptr, acc, mask=C_mask)


# Kernel: row-wise softmax with causal mask on vector V[N] -> S[N]
# Applies s[j] = exp(v[j] - max), divided by sum of exp over masked j. Mask: j >= pos.
@triton.jit
def softmax_mask_row_kernel(V_ptr, S_ptr, N, pos,
                            BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)  # program runs once per vector (here, per query head)
    offs = tl.arange(0, BLOCK_N)
    mask_n = offs < N
    v = tl.load(V_ptr + offs, mask=mask_n, other=-float("inf"))
    # Compute max for numerical stability
    max_v = tl.max(v, axis=0)
    v = v - max_v
    # Apply causal mask: j >= pos
    # Triton doesn't support dynamic indexing; create mask with broadcasting
    mask_j = offs >= pos
    v = tl.where(mask_j, v, -float("inf"))
    exp_v = tl.exp(v)
    sum_v = tl.sum(exp_v, axis=0)
    s = exp_v / sum_v
    tl.store(S_ptr + offs, s, mask=mask_n)


# Kernel: row-wise logsumexp in base-2 on masked vector V[N] -> scalar base-2 lse
# Applies lse = log(sum(exp(v - max))) / log(2) over masked positions (j >= pos).
@triton.jit
def lse_mask_base2_row_kernel(V_ptr, LSE_ptr, N, pos,
                              BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)  # program runs once per vector (here, per query head)
    offs = tl.arange(0, BLOCK_N)
    mask_n = offs < N
    v = tl.load(V_ptr + offs, mask=mask_n, other=-float("inf"))
    max_v = tl.max(v, axis=0)
    v = v - max_v
    mask_j = offs >= pos
    v = tl.where(mask_j, v, -float("inf"))
    exp_v = tl.exp(v)
    sum_v = tl.sum(exp_v, axis=0)
    lse_val = tl.log(sum_v) / 1.4426950408889634  # 1 / log(2)
    tl.store(LSE_ptr, lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Inputs are CUDA tensors from get_inputs(); ensure device
        device = q_nope.device

        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages = ckv_cache.shape[1]
        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1
        num_kv_indices = kv_indices.shape[0]

        # Constraints from original code
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64
        assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1

        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                continue

            kv_len = page_end - page_beg
            tok_idx = kv_indices[page_beg:page_end].to(torch.int64).to(device)

            # Prepare output row for this batch
            q_curr = q_start
            i = 0  # loop over queries in this batch
            while q_curr < q_end:
                i += 1

                # Copy qn and qp rows to fp32 buffers
                # A_qn: [16, 512], B_qn: [16, 16*512]
                A_qn = q_nope[q_curr]  # [16, 512], fp32
                A_qn_fp32 = A_qn.to(torch.float32)
                M_qn = 16
                K_qn = 512
                B_qn = torch.empty((1, M_qn * K_qn), dtype=torch.float32, device=device)
                # Launch Triton kernel once
                copy_row_to_fp32_kernel[(1,)](
                    A_qn_fp32, B_qn,
                    1, M_qn, K_qn,
                    A_qn_fp32.stride(0), A_qn_fp32.stride(1), A_qn_fp32.stride(2),
                    B_qn.stride(0), B_qn.stride(1),
                    BLOCK_M=16, BLOCK_K=512,
                    num_warps=4, num_stages=2
                )

                # A_qp: [16, 64], B_qp: [16, 16*64]
                A_qp = q_pe[q_curr]  # [16, 64], fp32
                A_qp_fp32 = A_qp.to(torch.float32)
                M_qp = 16
                K_qp = 64
                B_qp = torch.empty((1, M_qp * K_qp), dtype=torch.float32, device=device)
                copy_row_to_fp32_kernel[(1,)](
                    A_qp_fp32, B_qp,
                    1, M_qp, K_qp,
                    A_qp_fp32.stride(0), A_qp_fp32.stride(1), A_qp_fp32.stride(2),
                    B_qp.stride(0), B_qp.stride(1),
                    BLOCK_M=16, BLOCK_K=64,
                    num_warps=2, num_stages=2
                )

                # Gather cached keys for this batch element: tok_idx has L tokens
                # Kc_all rows: [num_pages, 1, 512] -> squeeze(1) gives [num_pages, 512]; we need tok_idx rows
                # Create A_Kc rows for each j in tok_idx
                Kc_rows = torch.empty((kv_len, head_dim_ckv), dtype=torch.float32, device=device)
                for j in range(kv_len):
                    idx = int(tok_idx[j].item())
                    A_Kc_j = ckv_cache[idx].squeeze(1).to(torch.float32)  # [512]
                    # Flatten to 2D [1, 512] for Triton copy
                    A_Kc_j = A_Kc_j.view(1, head_dim_ckv)
                    B_Kc_j = torch.empty((1, head_dim_ckv), dtype=torch.float32, device=device)
                    copy_row_to_fp32_kernel[(1,)](
                        A_Kc_j, B_Kc_j,
                        1, 1, head_dim_ckv,
                        A_Kc_j.stride(0), A_Kc_j.stride(1), A_Kc_j.stride(2),
                        B_Kc_j.stride(0), B_Kc_j.stride(1),
                        BLOCK_M=1, BLOCK_K=512,
                        num_warps=4, num_stages=2
                    )
                    # Place into Kc_rows[j]
                    Kc_rows[j] = B_Kc_j[0]

                # Kp rows similarly
                Kp_rows = torch.empty((kv_len, head_dim_kpe), dtype=torch.float32, device=device)
                for j in range(kv_len):
                    idx = int(tok_idx[j].item())
                    A_Kp_j = kpe_cache[idx].squeeze(1).to(torch.float32)  # [64]
                    A_Kp_j = A_Kp_j.view(1, head_dim_kpe)
                    B_Kp_j = torch.empty((1, head_dim_kpe), dtype=torch.float32, device=device)
                    copy_row_to_fp32_kernel[(1,)](
                        A_Kp_j, B_Kp_j,
                        1, 1, head_dim_kpe,
                        A_Kp_j.stride(0), A_Kp_j.stride(1), A_Kp_j.stride(2),
                        B_Kp_j.stride(0), B_Kp_j.stride(1),
                        BLOCK_M=1, BLOCK_K=64,
                        num_warps=4, num_stages=2
                    )
                    Kp_rows[j] = B_Kp_j[0]

                # Compute logits = (qn @ Kc.T) + (qp @ Kp.T) -> [16, kv_len]
                M_out = 16
                N = kv_len
                K1 = head_dim_ckv
                A_qn_view = B_qn.view(M_out, K1)  # [16, 512]
                # B^T for Kc: [N, K1] (we have Kc_rows [N, K1], flatten to [N, K1] already)
                KcT = Kc_rows.t().contiguous()  # [K1, N] then transpose doesn't apply since rows are [N, K1]
                # Instead, we directly use Kc_rows as [N, K1] by stacking rows
                B_KcT = torch.empty((N, K1), dtype=torch.float32, device=device)
                for j in range(N):
                    B_KcT[j] = Kc_rows[j]
                C_logits = torch.empty((M_out, N), dtype=torch.float32, device=device)
                matmul_left_kernel[(M_out, N,)](
                    A_qn_view, B_KcT, C_logits,
                    M_out, N, K1,
                    A_qn_view.stride(0), A_qn_view.stride(1), 0,  # K1 stride handled by N
                    B_KcT.stride(0), B_KcT.stride(1), C_logits.stride(0), C_logits.stride(1),
                    BLOCK_M=16, BLOCK_N=N, BLOCK_K=32,
                    num_warps=4, num_stages=2
                )

                # Compute (qp @ Kp.T): [16, 64] @ [64, N] -> [16, N]
                A_qp_view = B_qp.view(M_out, head_dim_kpe)  # [16, 64]
                B_KpT = torch.empty((N, head_dim_kpe), dtype=torch.float32, device=device)
                for j in range(N):
                    B_KpT[j] = Kp_rows[j]
                C_qpKp = torch.empty((M_out, N), dtype=torch.float32, device=device)
                matmul_left_kernel[(M_out, N,)](
                    A_qp_view, B_KpT, C_qpKp,
                    M_out, N, head_dim_kpe,
                    A_qp_view.stride(0), A_qp_view.stride(1), 0,
                    B_KpT.stride(0), B_KpT.stride(1), C_qpKp.stride(0), C_qpKp.stride(1),
                    BLOCK_M=16, BLOCK_N=N, BLOCK_K=32,
                    num_warps=4, num_stages=2
                )

                logits = C_logits + C_qpKp  # [16, N]
                logits = logits * sm_scale

                # Compute causal mask pos = L - (q_end - q_start) + i; here q_end - q_start = 1 so q_len=1
                # We set pos = L - 1 + i; note: i here is 1, so pos = L
                pos = (kv_len - 1) + 1  # since i=1, we apply mask j >= pos; pos = L
                N_out = N
                # Row-wise softmax with mask
                S_softmax = torch.empty((N_out,), dtype=torch.float32, device=device)
                softmax_mask_row_kernel[(1,)](
                    logits[0], S_softmax, N_out, pos,
                    BLOCK_N=32,
                    num_warps=2, num_stages=2
                )

                # Compute logsumexp base-2 (masked)
                LSE_vec = torch.empty((1,), dtype=torch.float32, device=device)
                lse_mask_base2_row_kernel[(1,)](
                    logits[0], LSE_vec, N_out, pos,
                    BLOCK_N=32,
                    num_warps=2, num_stages=2
                )

                # Compute output: softmax @ Kc
                # We need Kc rows for each j; create B^T for Kc (KcT: [N, K1])
                B_KcT = torch.empty((N, K1), dtype=torch.float32, device=device)
                for j in range(N):
                    B_KcT[j] = Kc_rows[j]  # [N, K1]
                C_out = torch.empty((M_out, K1), dtype=torch.float32, device=device)
                matmul_left_kernel[(M_out, K1,)](
                    S_softmax.view(1, N), B_KcT, C_out,
                    M_out, N, K1,
                    S_softmax.view(1, N).stride(0), S_softmax.view(1, N).stride(1), 0,
                    B_KcT.stride(0), B_KcT.stride(1), C_out.stride(0), C_out.stride(1),
                    BLOCK_M=16, BLOCK_N=N, BLOCK_K=64,
                    num_warps=4, num_stages=2
                )

                # Store output to [q_start + i, :, :]
                # Cast to bfloat16 for output
                out_row = C_out[0].to(torch.bfloat16).view(num_qo_heads, head_dim_ckv)
                # lse stores scalar per head; we have single head for this row
                lse[q_curr] = LSE_vec[0]

                # Move to next query in this batch
                q_curr += 1

        return output, lse


def run(*args):
    return ModelNew()(*args)
