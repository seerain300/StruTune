import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 1: Compute logits_scaled = (qn @ Kc.T) + (qp @ Kp.T) for a single (b, h)
# Inputs:
#   qn_ptr: [Hc] float32 (row vector for head h from q_nope[b])
#   qp_ptr: [Hp] float32 (row vector for head h from q_pe[b])
#   Kc_ptr: [L, Hc] float32
#   Kp_ptr: [L, Hp] float32
#   out_logit_ptr: [L] float32
#   sm_scale: float32
#   Hc: int
#   Hp: int
#   L: int
@triton.jit
def matmul_add_row_kernel(
    qn_ptr, qp_ptr, Kc_ptr, Kp_ptr, out_logit_ptr,
    sm_scale, Hc, Hp, L
):
    # Single program computes logits for all tokens for one (b, h)
    # We iterate over tokens k and accumulate qn @ Kc[k, :] + qp @ Kp[k, :]
    # Note: Hc, Hp, L are runtime ints here; Triton loops over them.
    for k in range(0, L):
        sum_qn = 0.0
        sum_qp = 0.0
        # Reduce over Hc and Hp
        for i in range(0, Hc):
            sum_qn += qn_ptr[i] * tl.load(Kc_ptr + k * Hc + i)
        for j in range(0, Hp):
            sum_qp += qp_ptr[j] * tl.load(Kp_ptr + k * Hp + j)
        out_val = sum_qn + sum_qp
        out_val = out_val * sm_scale
        tl.store(out_logit_ptr + k, out_val)


# Kernel 2: Row-wise softmax + logsumexp (lse) for a single (b, h)
# Inputs:
#   logits_ptr: [L] float32
#   out_lse_ptr: scalar float32
#   attn_ptr: [L] float32 (will store softmax)
#   L: int
# This kernel does:
#   pass1: max over logits
#   pass2: sum of exp(logits - max)
#   compute lse = log(sum) / log(2.0)
#   pass3: write normalized attn = exp(logits - max) / sum
@triton.jit
def softmax_logsumexp_row_kernel(
    logits_ptr, out_lse_ptr, attn_ptr, L
):
    # Compute max
    max_val = -1e30
    for i in range(0, L):
        max_val = tl.maximum(max_val, logits_ptr[i])
    # Compute sum of exp(logits - max)
    sum_exp = 0.0
    for i in range(0, L):
        sum_exp += tl.exp(logits_ptr[i] - max_val)
    lse = tl.log(sum_exp) / 2.0  # log2(sum) = ln(sum) / ln(2)
    tl.store(out_lse_ptr, lse)
    # Compute and store softmax (attention)
    inv_sum = 1.0 / sum_exp
    for i in range(0, L):
        attn_ptr[i] = tl.exp(logits_ptr[i] - max_val) * inv_sum


# Kernel 3: Matvec GEMV for a single output chunk over tokens for a given (b, h)
# Input:
#   attn_ptr: [L] float32 (softmax attention for that head)
#   Kc_ptr: [L, Hc] float32
#   out_ptr: [Hc_chunk] float32
#   Hc: int (output dim)
#   L: int (token count)
#   BLOCK_N: number of output columns handled by this program
@triton.jit
def matvec_row_kernel(
    attn_ptr, Kc_ptr, out_ptr, Hc, L, BLOCK_N
):
    # Each program handles a chunk of output columns [start, start+BLOCK_N)
    start = tl.program_id(0) * BLOCK_N
    offs = start + tl.arange(0, BLOCK_N)
    mask = offs < Hc
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    # Loop over tokens and accumulate
    for k in range(0, L):
        val = attn_ptr[k]
        # Row k of Kc is contiguous across Hc: Kc[k, offs] = Kc_ptr + k*Hc + offs
        row_ptr = Kc_ptr + k * Hc + offs
        acc += val * tl.load(row_ptr, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA."
        device = q_nope.device

        batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
        _, _, head_dim_kpe = q_pe.shape
        num_pages = ckv_cache.shape[0]
        assert ckv_cache.shape == (num_pages, 1, head_dim_ckv)
        assert kpe_cache.shape == (num_pages, 1, head_dim_kpe)

        # Prepare Kc_all and Kp_all (float32 for stable math)
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, head_dim_ckv]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, head_dim_kpe]

        # Compute tok_idx for each batch b using kv_indptr and kv_indices (as in the original)
        # kv_indptr shape: [batch_size+1], kv_indices shape: [num_kv_indices]
        # tok_idx[b] = kv_indices[page_beg:b : page_end]
        # len_indptr = kv_indptr.shape[0] == batch_size + 1
        len_indptr = kv_indptr.shape[0]
        assert len_indptr == batch_size + 1, "kv_indptr length must be batch_size + 1"

        # Allocate output and lse
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # For each batch element
        for b in range(batch_size):
            # derive token range for this batch
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L_tokens = max(0, page_end - page_beg)
            if L_tokens == 0:
                # No KV tokens for this batch element: output zeros and lse -inf
                output[b] = torch.zeros((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
                lse[b] = torch.full((num_qo_heads,), -float('inf'), dtype=torch.float32, device=device)
                continue

            tok_idx = (kv_indices[page_beg:page_end]).to(torch.int32).to(device)
            # Gather Kc and Kp for these tokens
            Kc = Kc_all[tok_idx].contiguous().to(torch.float32)  # [L_tokens, head_dim_ckv]
            Kp = Kp_all[tok_idx].contiguous().to(torch.float32)  # [L_tokens, head_dim_kpe]

            # Prepare per-head qn and qp
            # q_nope: [B, H, Hc], q_pe: [B, H, Hp]
            qn = q_nope[b].contiguous().to(torch.float32)  # [H, Hc] -> use only row h
            # Note: num_qo_heads is H dimension. We compute per head h.
            # We will loop over heads h and launch kernels for each.

            # For each head h
            for h in range(num_qo_heads):
                # 1) Compute logits_scaled for this head
                logits = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                # Launch GEMV kernel: one program per (b,h) reducing over tokens and dims
                # Pass Hc, Hp, L as runtime ints
                matmul_add_row_kernel[(1,)](
                    qn[h, :].contiguous(), q_pe[b, h, :].contiguous(), Kc, Kp, logits,
                    sm_scale, head_dim_ckv, head_dim_kpe, L_tokens,
                    num_warps=1, num_stages=1
                )
                # 2) Compute softmax (attention) and lse for this head
                attn = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                softmax_logsumexp_row_kernel[(1,)](
                    logits, lse[b, h], attn,
                    L_tokens,
                    num_warps=1, num_stages=1
                )
                # 3) Compute output[b, h, :] = attn @ Kc
                out_row = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
                BLOCK_N = 128  # handle up to 512 dims in chunks
                grid = (triton.cdiv(head_dim_ckv, BLOCK_N),)
                matvec_row_kernel[grid](
                    attn, Kc, out_row, head_dim_ckv, L_tokens, BLOCK_N,
                    num_warps=2, num_stages=2
                )
                # Store out_row into output[b, h, :]
                output[b, h, :] = out_row

        # Return output in bfloat16 (original dtype), and lse as float32
        return output.to(torch.bfloat16), lse


# Optional helpers for testing (not used by evaluator, but provided for parity)
def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to(device='cuda')
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32).to(device='cuda')
    sm_scale = 1.0
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    return ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)