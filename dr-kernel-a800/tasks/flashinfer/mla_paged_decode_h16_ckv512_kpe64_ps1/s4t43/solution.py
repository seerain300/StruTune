import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_row_kernel(
    qn_ptr,         # *float32, [Kc_dim] contiguous
    Kc_ptr,         # *float32, [num_tokens, Kc_dim] contiguous
    qp_ptr,         # *float32, [Kp_dim] contiguous
    Kp_ptr,         # *float32, [num_tokens, Kp_dim] contiguous
    logits_ptr,     # *float32, [num_tokens] contiguous
    num_tokens,     # int, runtime M
    Kc_dim: tl.constexpr,      # 512
    Kp_dim: tl.constexpr,      # 64
    BLOCK_K: tl.constexpr = 64
):
    i = tl.program_id(0)
    if i >= num_tokens:
        return
    acc = 0.0
    # Accumulate dot(qn, Kc[i, :])
    for k0 in range(0, Kc_dim, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        mask = offs < Kc_dim
        a = tl.load(qn_ptr + offs, mask=mask, other=0.0)
        b = tl.load(Kc_ptr + i * Kc_dim + offs, mask=mask, other=0.0)
        acc += tl.sum(a * b, axis=0)
    # Accumulate dot(qp, Kp[i, :])
    acc_qp = 0.0
    for k0 in range(0, Kp_dim, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        mask = offs < Kp_dim
        a = tl.load(qp_ptr + offs, mask=mask, other=0.0)
        b = tl.load(Kp_ptr + i * Kp_dim + offs, mask=mask, other=0.0)
        acc_qp += tl.sum(a * b, axis=0)
    acc += acc_qp
    tl.store(logits_ptr + i, acc)


@triton.jit
def compute_softmax_lse_kernel(
    logits_ptr,         # *float32, [num_tokens]
    sm_scale,           # float32
    max_ptr,            # *float32, scalar [1]
    sum_ptr,            # *float32, scalar [1]
    num_tokens          # int, runtime
):
    # Compute max over logits * sm_scale
    max_val = -float("inf")
    i = 0
    while i < num_tokens:
        li = tl.load(logits_ptr + i)
        val = li * sm_scale
        if val > max_val:
            max_val = val
        i += 1
    tl.store(max_ptr, max_val)

    # Compute sum of exp(logits * sm_scale - max_val)
    sum_val = 0.0
    i = 0
    while i < num_tokens:
        li = tl.load(logits_ptr + i)
        val = li * sm_scale
        sum_val += tl.exp(val - max_val)
        i += 1
    tl.store(sum_ptr, sum_val)

    # lse = log(sum_exp) * sm_scale + max_val * sm_scale, divided by ln(2)
    # Note: Triton can use tl.log. We compute and write only if needed; in host we can combine.
    # We'll store sum_ptr for host to combine, but here we write lse via host function ModelNew.forward.
    # Not stored; host will compute via Triton-accelerated softmax and write lse scalar.


@triton.jit
def write_attn_kernel(
    logits_ptr,     # *float32, [num_tokens]
    sm_scale,       # float32
    max_ptr,        # *float32, scalar [1]
    sum_ptr,        # *float32, scalar [1]
    attn_ptr,       # *float32, [num_tokens] contiguous
    num_tokens      # int, runtime
):
    max_val = tl.load(max_ptr)
    sum_exp = tl.load(sum_ptr)
    i = 0
    while i < num_tokens:
        li = tl.load(logits_ptr + i)
        val = li * sm_scale
        attn_i = tl.exp(val - max_val) / sum_exp
        tl.store(attn_ptr + i, attn_i)
        i += 1


@triton.jit
def matvec_out_row_kernel(
    attn_ptr,       # *float32, [num_tokens]
    Kc_ptr,         # *float32, [num_tokens, Kc_dim]
    out_ptr,        # *float32, [Kc_dim]
    num_tokens,     # int
    Kc_dim: tl.constexpr,      # 512
    BLOCK_M: tl.constexpr = 64
):
    k_out = tl.program_id(0)  # output dimension in 0..Kc_dim-1
    acc = 0.0
    for m0 in range(0, num_tokens, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < num_tokens
        attn_chunk = tl.load(attn_ptr + offs_m, mask=mask_m, other=0.0)
        # Accumulate Kc rows for these offs_m across k_out
        for j in range(0, Kc_dim, BLOCK_M):
            offs_j = j + tl.arange(0, BLOCK_M)
            mask_j = offs_j < Kc_dim
            # Load Kc rows: Kc_ptr is [num_tokens, Kc_dim], rows are offs_m, cols offs_j
            Kc_chunk = tl.zeros([BLOCK_M], dtype=tl.float32)
            # We need to load Kc[offs_m, offs_j]; vectorized over offs_m and offs_j
            # Implement as nested loops since Triton can handle while loops:
            for mm in range(0, BLOCK_M):
                m = m0 + mm
                m_valid = m < num_tokens
                # if m_valid: Kc_chunk[mm] = sum over jj of attn_chunk[mm] * Kc[m, jj]
                # but we can directly load per m
                if m_valid:
                    # Loop over offs_j (constexpr range)
                    for jj in range(0, Kc_dim, BLOCK_M):
                        jj_offs = jj + tl.arange(0, BLOCK_M)
                        jj_mask = jj_offs < Kc_dim
                        Kcol = tl.load(Kc_ptr + m * Kc_dim + jj_offs, mask=jj_mask, other=0.0)
                        Kc_chunk[mm] += tl.sum(attn_ptr[m] * Kcol, axis=0)
            # Now add to acc[k_out] contribution: Kc_chunk[k_out]
            acc += Kc_chunk[k_out]
    tl.store(out_ptr + k_out, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes from original code
        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        Kc_dim = q_nope.shape[2]  # 512
        Kp_dim = q_pe.shape[2]    # 64
        num_tokens = kv_indptr.shape[0] - 1
        device = q_nope.device

        # Prepare output tensors
        output = torch.empty(
            (batch_size, num_qo_heads, Kc_dim), dtype=torch.float32, device=device
        )
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Loop over batches
        for b in range(batch_size):
            # Gather tokens for this batch from kv_indices and kv_indptr
            # tokens range [kv_indptr[b]: kv_indptr[b+1])
            if num_tokens <= 0:
                # No tokens; output zeros, lse can be zero
                output[b] = 0.0
                lse[b] = 0.0
                continue

            # Compute M for this batch
            M = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())

            # Gather Kc rows and Kp rows
            tokens = kv_indices[int(kv_indptr[b]):int(kv_indptr[b + 1])]
            Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
            Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]
            Kc = Kc_all[tokens]  # [M, 512]
            Kp = Kp_all[tokens]  # [M, 64]

            # Allocate per-batch tensors
            logits = torch.empty((M,), dtype=torch.float32, device=device)

            # Kernel 1: compute logits per token (use Triton, one program per token)
            grid_logits = (M,)
            # Convert q_nope and q_pe to float32 contiguous vectors for head h=0
            # We need per-head vectors; since original asserts num_qo_heads=16 and q_nope shape (B,16,512), we can proceed:
            # For Triton, we pass pointers to current batch's heads. Since Triton kernels here are scalar per-batch,
            # we'll compute for all heads in host loops and reuse kernels.
            for h in range(num_qo_heads):
                qn = q_nope[b, h].to(torch.float32).contiguous()  # [512]
                qp = q_pe[b, h].to(torch.float32).contiguous()   # [64]
                compute_logits_row_kernel[grid_logits](
                    qn, Kc, qp, Kp, logits, M, Kc_dim, Kp_dim, BLOCK_K=64
                )

                # Kernel 2: compute max and sum_exp for lse
                max_val = torch.empty((1,), dtype=torch.float32, device=device)
                sum_val = torch.empty((1,), dtype=torch.float32, device=device)
                compute_softmax_lse_kernel[(1,)](
                    logits, sm_scale, max_val, sum_val, M
                )
                # Compute lse[h] = log(sum_exp) * sm_scale + max_val * sm_scale, divided by ln(2)
                lse[b, h] = torch.log(sum_val) * sm_scale + max_val * (sm_scale / math.log(2.0))

                # Kernel 3: write attn vector
                attn = torch.empty((M,), dtype=torch.float32, device=device)
                write_attn_kernel[(M,)](
                    logits, sm_scale, max_val, sum_val, attn, M
                )

                # Kernel 4: matvec to compute out[h, :]
                out_vec = torch.empty((Kc_dim,), dtype=torch.float32, device=device)
                grid_out = (Kc_dim,)
                matvec_out_row_kernel[grid_out](
                    attn, Kc, out_vec, M, Kc_dim, BLOCK_M=64
                )

                # Store output for this head
                output[b, h] = out_vec

            # Cast output to bfloat16 to match original function
            output[b] = output[b].to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
