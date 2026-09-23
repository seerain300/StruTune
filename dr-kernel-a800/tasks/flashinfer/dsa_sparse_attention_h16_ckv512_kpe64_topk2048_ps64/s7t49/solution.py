import torch
import triton
import triton.language as tl

@triton.jit
def gather_rows_kernel(
    idx_ptr,               # *int32, shape [L]
    src_ptr,               # *fp32,  flattened source [num_rows * head_dim]
    dst_ptr,               # *fp32,  destination [L * head_dim]
    row_stride,            # int32,  number of rows per token
    head_dim: tl.constexpr,# int,    e.g., 512 for CKV, 64 for KPE
    L: tl.constexpr,       # int,    number of valid indices
):
    # One program instance handles one destination row
    pid = tl.program_id(0)  # row index in destination [0..L)
    if pid >= L:
        return

    # Load index for this row
    idx = tl.load(idx_ptr + pid)  # int32

    # Compute source row offset: idx * row_stride
    src_row = idx * row_stride

    # Copy one column at a time from src to dst
    for c in range(0, head_dim):
        val = tl.load(src_ptr + src_row * head_dim + c)
        tl.store(dst_ptr + pid * head_dim + c, val)


@triton.jit
def per_token_attention_kernel(
    qn_ptr,                # *fp32,  q_nope[t, h, :] flattened [512]
    qp_ptr,                # *fp32,  q_pe[t, h, :] flattened [64]
    sparse_ptr,            # *int32, sparse_indices[t, :] flattened [topk]
    Kc_all_ptr,            # *fp32,  flattened [num_pages*64, 512]
    Kp_all_ptr,            # *fp32,  flattened [num_pages*64, 64]
    out_row_ptr,           # *fp32,  output row [512]
    lse_ptr,               # *fp32,  scalar [1] for lse of this (t,h)
    sm_scale: tl.float32,  # scaling factor
    row_stride: tl.int32,  # num_rows per token (64)
    head_dim_kc: tl.constexpr,  # 512
    head_dim_kp: tl.constexpr,  # 64
    topk: tl.constexpr,         # 2048
    num_valid: tl.constexpr,    # number of valid indices in sparse_indices[t, :], typically <= topk
):
    # This kernel processes a single (token, head) pair
    # It gathers Kc_t and Kp_t using gather_rows_kernel, then computes attention for that head.

    # Work variables
    sum_exp = 0.0
    m = -float("inf")

    # Gather Kc_t and Kp_t: we will construct them as temporary buffers of size [num_valid * head_dim]
    Kc_t = tl.zeros([num_valid * head_dim_kc], dtype=tl.float32)
    Kp_t = tl.zeros([num_valid * head_dim_kp], dtype=tl.float32)

    # First, populate Kc_t and Kp_t by looping over valid indices and calling gather_rows_kernel
    # We will launch gather_rows_kernel in chunks for correctness. However, Triton requires
    # compile-time static shapes; to keep it simple and safe, we process each valid index sequentially
    # using a for loop. Note: num_valid is a tl.constexpr, so this is supported here.

    for i in range(0, num_valid):
        # Build index arrays of length 1
        idx_i = sparse_ptr + i  # pointer to int32 at position i in sparse row
        # Destination buffers for this i
        dstKc_i = Kc_t + i * head_dim_kc
        dstKp_i = Kp_t + i * head_dim_kp
        # Launch gather_rows_kernel to fill these slices
        gather_rows_kernel[(1,)](
            idx_i,          # idx_ptr (single int32)
            Kc_all_ptr,     # src_ptr
            dstKc_i,        # dst_ptr
            row_stride,     # row_stride
            head_dim_kc,    # head_dim
            1,              # L (single row)
        )
        gather_rows_kernel[(1,)](
            idx_i,          # idx_ptr (single int32)
            Kp_all_ptr,     # src_ptr
            dstKp_i,        # dst_ptr
            row_stride,     # row_stride
            head_dim_kp,    # head_dim
            1,              # L (single row)
        )

    # Now compute attention for this (token, head)
    # We iterate over keys in chunks to maintain scalar m and sum_exp.
    # However, since num_valid is small (<= 2048), we can iterate directly.
    for j in range(0, num_valid):
        # Load qn_row and qp_row (they are scalars via flattened pointers)
        # qn_ptr is already [512] for this head; we iterate over c to compute dot
        # We'll compute logits for this key j: sum_c qn[c] * Kc_t[j*512 + c] + sum_c qp[c] * Kp_t[j*64 + c]
        # But Triton cannot index with dynamic expressions easily here; instead, compute dot via sum over columns.
        dot_ckv = 0.0
        dot_kpe = 0.0
        for c in range(0, head_dim_kc):
            qn_c = tl.load(qn_ptr + c)
            kc_c = tl.load(Kc_t + j * head_dim_kc + c)
            dot_ckv += qn_c * kc_c
        for c in range(0, head_dim_kp):
            qp_c = tl.load(qp_ptr + c)
            kp_c = tl.load(Kp_t + j * head_dim_kp + c)
            dot_kpe += qp_c * kp_c

        logits_j = dot_ckv + dot_kpe
        logits_scaled = logits_j * sm_scale
        m_new = tl.maximum(m, logits_scaled)
        sum_exp = sum_exp * tl.exp(m - m_new) + tl.exp(logits_scaled - m_new)
        m = m_new

    # Compute lse in base-2 (original code uses logsumexp / ln(2))
    inv_ln2 = 1.4426950408889634  # 1 / ln(2)
    lse_val = m + tl.log(sum_exp) * inv_ln2
    tl.store(lse_ptr, lse_val)

    # Compute output for this head: out_row = sum_j softmax_j * Kc_t[j, :]
    # We need to recompute softmax contributions. For each j:
    for j in range(0, num_valid):
        # Softmax contribution for this j
        dot_ckv = 0.0
        for c in range(0, head_dim_kc):
            qn_c = tl.load(qn_ptr + c)
            kc_c = tl.load(Kc_t + j * head_dim_kc + c)
            dot_ckv += qn_c * kc_c
        dot_kpe = 0.0
        for c in range(0, head_dim_kp):
            qp_c = tl.load(qp_ptr + c)
            kp_c = tl.load(Kp_t + j * head_dim_kp + c)
            dot_kpe += qp_c * kp_c
        logits_j = dot_ckv + dot_kpe
        logits_scaled = logits_j * sm_scale
        soft_j = tl.exp(logits_scaled - lse_val * inv_ln2)  # softmax scaled by lse
        # Accumulate output: out_row += soft_j * Kc_t[j, :]
        for c in range(0, head_dim_kc):
            kc_c = tl.load(Kc_t + j * head_dim_kc + c)
            out_c = tl.load(out_row_ptr + c)
            out_c += soft_j * kc_c
            tl.store(out_row_ptr + c, out_c)

@triton.jit
def write_output_kernel(
    out_row_ptr,           # *fp32,  flattened output row [512]
    dst_out_ptr,           # *fp32,  destination out[t, h, :] flattened [512]
):
    pid = tl.program_id(0)
    # Write out_row_ptr to dst_out_ptr at position pid
    # pid is 0 because we launch 1 program instance per head
    for c in range(0, 512):
        val = tl.load(out_row_ptr + c)
        tl.store(dst_out_ptr + c, val)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        q_nope,     # [num_tokens, num_qo_heads, 512], bfloat16
        q_pe,       # [num_tokens, num_qo_heads, 64],  bfloat16
        ckv_cache,  # [num_pages, 64, 512], bfloat16
        kpe_cache,  # [num_pages, 64, 64],  bfloat16
        sparse_indices,  # [num_tokens, topk], int32 (topk=2048)
        sm_scale,               # float32 scalar
    ):
        # Ensure on CUDA
        device = q_nope.device

        # Prepare flattened caches in float32
        Kc_all = ckv_cache.reshape(-1, 512).to(torch.float32).contiguous()  # [num_pages*64, 512]
        Kp_all = kpe_cache.reshape(-1, 64).to(torch.float32).contiguous()   # [num_pages*64, 64]

        num_tokens = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]

        # Output buffer: store float32 in kernel, then cast to bfloat16 at the end
        output = torch.empty((num_tokens, num_qo_heads, 512), dtype=torch.float32, device=device)
        lse = torch.empty((num_tokens, num_qo_heads), dtype=torch.float32, device=device)

        # We process one (token, head) per program instance
        grid = (num_tokens, num_qo_heads)

        for t in range(num_tokens):
            sparse_row = sparse_indices[t].contiguous()  # [topk], int32
            # Determine number of valid indices: count non -1 entries
            # Since num_valid is runtime-dependent and Triton needs constexpr, we pass it as a kernel parameter.
            # Compute num_valid on host and pass as scalar.
            num_valid = int((sparse_row != -1).sum().item())

            # Prepare qn and qp as flattened vectors for this head
            # q_nope[t, h, :] and q_pe[t, h, :]
            for h in range(num_qo_heads):
                qn_vec = q_nope[t, h, :].to(torch.float32).contiguous()  # [512]
                qp_vec = q_pe[t, h, :].to(torch.float32).contiguous()    # [64]

                # Allocate output row for this head
                out_row = torch.empty((512,), dtype=torch.float32, device=device)

                # Launch per-token attention kernel: computes lse and fills out_row
                per_token_attention_kernel[grid](
                    qn_vec,                    # qn_ptr
                    qp_vec,                    # qp_ptr
                    sparse_row,                # sparse_ptr (int32)
                    Kc_all,                    # Kc_all_ptr
                    Kp_all,                    # Kp_all_ptr
                    out_row,                   # out_row_ptr
                    lse[t, h],                 # lse_ptr (scalar)
                    sm_scale,                  # sm_scale
                    64,                        # row_stride (64 rows per token)
                    512,                       # head_dim_kc
                    64,                        # head_dim_kp
                    2048,                      # topk
                    num_valid,                 # num_valid (constexpr)
                    num_warps=1,
                )

                # Write output row to output[t, h, :]
                dst_ptr = output[t, h, :].contiguous()
                write_output_kernel[(1,)](out_row, dst_ptr)

        # Cast output to bfloat16 to match original signature
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
