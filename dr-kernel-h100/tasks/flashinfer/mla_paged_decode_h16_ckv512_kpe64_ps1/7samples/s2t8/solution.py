import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _single_head_compute(
    q_nope_ptr, q_pe_ptr,
    Kc_rows_ptr, Kp_rows_ptr,
    out_ptr, lse_ptr,
    B: tl.constexpr, H: tl.constexpr,
    Dc: tl.constexpr, Dp: tl.constexpr,
    L_tokens: tl.constexpr,
    sm_scale: tl.float32
):
    # program ids: each program handles one (b, h)
    b = tl.program_id(0)
    h = tl.program_id(1)

    # base offsets for q_nope[b, h, :] and q_pe[b, h, :]
    qn_base = b * (H * Dc) + h * Dc
    qp_base = b * (H * Dp) + h * Dp

    # Prepare logits vector for L_tokens tokens
    logits = tl.zeros((L_tokens,), dtype=tl.float32)

    # Compute dot products: qn @ Kc.T and qp @ Kp.T
    # Kc_rows_ptr and Kp_rows_ptr are 2D views of size [L_tokens, Dc] and [L_tokens, Dp]
    # We access rows via pointer math: row t at offset t * dim + j for dimension j
    # First accumulate qn @ Kc.T
    for i in tl.static_range(0, Dc):
        qni = tl.load(q_nope_ptr + qn_base + i)  # float32 scalar
        for t in tl.static_range(0, L_tokens):
            Kc_row_t = Kc_rows_ptr + t * Dc  # base pointer to row t in Kc_rows
            Kc_t_i = tl.load(Kc_row_t + i)   # float32 scalar
            logits[t] += qni * Kc_t_i

    # Then accumulate qp @ Kp.T
    for j in tl.static_range(0, Dp):
        qpj = tl.load(q_pe_ptr + qp_base + j)  # float32 scalar
        for t in tl.static_range(0, L_tokens):
            Kp_row_t = Kp_rows_ptr + t * Dp  # base pointer to row t in Kp_rows
            Kp_t_j = tl.load(Kp_row_t + j)   # float32 scalar
            logits[t] += qpj * Kp_t_j

    # Scale logits
    logits_scaled = logits * sm_scale

    # Numerically stable logsumexp in base 2: lse = (log(sum_exp) + |max|) / ln(2)
    max_logit = -float("inf")
    for t in tl.static_range(0, L_tokens):
        max_logit = tl.maximum(max_logit, logits_scaled[t])

    sum_exp = 0.0
    for t in tl.static_range(0, L_tokens):
        sum_exp += tl.exp(logits_scaled[t] - max_logit)

    lse = (tl.log(sum_exp) + max_logit) / tl.log(2.0)
    tl.store(lse_ptr + b * H + h, lse)  # store per-head lse for batch b

    # Compute softmax attn
    attn = tl.zeros((L_tokens,), dtype=tl.float32)
    ln2 = 0.6931471805599453  # 1 / ln(2)
    for t in tl.static_range(0, L_tokens):
        attn[t] = tl.exp(logits_scaled[t] - lse) / ln2

    # Final output vector: out[b, h, :] = sum_t attn[t] * Kc[t, :]
    out_base = (b * H + h) * Dc
    for i in tl.static_range(0, Dc):
        acc = 0.0
        for t in tl.static_range(0, L_tokens):
            Kc_row_t = Kc_rows_ptr + t * Dc
            Kc_t_i = tl.load(Kc_row_t + i)
            acc += attn[t] * Kc_t_i
        # Store as float32; host will convert to bfloat16 if needed
        tl.store(out_ptr + out_base + i, acc)


# Entry point ModelNew with Triton-only forward
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure device is CUDA for Triton
        device = q_nope.device
        assert device.type == "cuda", "Triton kernels require CUDA tensors"

        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]  # 512
        head_dim_kpe = q_pe.shape[2]    # 64

        # Prepare output and lse
        out = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Compute L_tokens per batch element from kv_indptr
        L_tokens_list = []
        for b in range(batch_size):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())  # assuming len_indptr == batch_size + 1
            L_tokens = end - start
            L_tokens_list.append(L_tokens)

        # Prepare Kc_rows and Kp_rows per batch element on device
        # Kc_all: [num_pages, Dc], kpe_cache: [num_pages, Dp]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, Dc]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, Dp]

        # For each batch element, gather Kc and Kp rows corresponding to tokens [start:end)
        for b in range(batch_size):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            L_tokens = end - start
            # indices for this batch element
            tok_idx = kv_indices[start:end].to(torch.int32)  # [L_tokens]
            # Gather rows
            Kc_rows = Kc_all[tok_idx]  # [L_tokens, Dc], float32
            Kp_rows = Kp_all[tok_idx]  # [L_tokens, Dp], float32

            # Launch Triton kernel for each head h
            grid = (1, num_qo_heads)
            _single_head_compute[grid](
                q_nope[b], q_pe[b],
                Kc_rows, Kp_rows,
                out[b], lse[b],
                batch_size, num_qo_heads,
                head_dim_ckv, head_dim_kpe,
                L_tokens,
                sm_scale
            )

        # Cast output to bfloat16 to match original return type
        out_bf16 = out.to(torch.bfloat16)
        return out_bf16, lse


def run(*args):
    return ModelNew()(*args)
