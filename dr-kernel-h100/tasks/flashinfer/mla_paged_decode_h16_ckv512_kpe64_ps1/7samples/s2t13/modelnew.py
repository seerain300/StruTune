import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute output and lse for a single (batch element b, head h).
# Grid is (B, H). Each program handles one b,h pair.
@triton.jit
def _batch_elem_kernel(
    q_nope_ptr, q_pe_ptr,
    Kc_all_ptr, Kp_all_ptr,
    out_ptr, lse_ptr,
    B: tl.int32, H: tl.int32,  # runtime ints
    Dc: tl.int32, Dp: tl.int32,  # runtime ints
    L_tokens: tl.int32,          # runtime int (number of tokens for this batch element)
    sm_scale: tl.float32          # runtime float
):
    b = tl.program_id(0)  # batch index
    h = tl.program_id(1)  # head index

    # We'll assume fixed sizes from the model: H=16, Dc=512, Dp=64
    # The kernel loops over tokens and dimensions using tl.static_range by passing these as tl.constexpr in launcher.
    # To keep signature simple, we use runtime ints and let Triton handle the loop indexing via Python launch values.

    # Compute base offsets for q_nope[b, h, :] and q_pe[b, h, :]
    # q_nope has shape [B, H, Dc], q_pe has shape [B, H, Dp]
    # We need q_nope_ptr + b * (H * Dc) + h * Dc
    base_qn = b * H * Dc + h * Dc
    base_qp = b * H * Dp + h * Dp

    # Vector for logits, lse, and attn
    # We'll compute logits_scaled, then lse, then attn, and finally out[h, :]
    logits = tl.zeros([L_tokens], dtype=tl.float32)
    lse = tl.full([1], -float('inf'), dtype=tl.float32)[0]

    # Compute logits[h, t] = qn @ Kc[t, :] + qp @ Kp[t, :]
    # Kc_all_ptr has shape [num_tokens, Dc], Kp_all_ptr has shape [num_tokens, Dp]
    # Loop over tokens
    for t in tl.static_range(0, L_tokens):
        sum_qn = 0.0
        sum_qp = 0.0
        # Accumulate qn @ Kc[t, :]
        for i in tl.static_range(0, Dc):
            sum_qn += tl.load(q_nope_ptr + base_qn + i) * tl.load(Kc_all_ptr + t * Dc + i)
        # Accumulate qp @ Kp[t, :]
        for j in tl.static_range(0, Dp):
            sum_qp += tl.load(q_pe_ptr + base_qp + j) * tl.load(Kp_all_ptr + t * Dp + j)
        logits[t] = sum_qn + sum_qp

    # Scale logits
    logits_scaled = logits * sm_scale

    # Stable logsumexp: lse = max + log(sum(exp(logits_scaled - max)))
    max_val = tl.max(logits_scaled, axis=0)
    sum_exp = 0.0
    for t in tl.static_range(0, L_tokens):
        sum_exp += tl.exp(logits_scaled[t] - max_val)
    lse = max_val + tl.log(sum_exp) / tl.log(2.0)

    # attn = exp(logits_scaled - lse) / ln(2)
    inv_ln2 = 1.4426950408889634  # 1 / ln(2)
    for t in tl.static_range(0, L_tokens):
        attn_t = tl.exp(logits_scaled[t] - lse) / inv_ln2

        # out[h, :] += attn_t * Kc[t, :]
        # Output vector for this head is out_ptr + b * H * Dc + h * Dc
        for i in tl.static_range(0, Dc):
            val = tl.load(Kc_all_ptr + t * Dc + i)
            # out_ptr has shape [B, H, Dc] with contiguous layout; we store as bfloat16
            out_ptr_val = tl.load(out_ptr + b * H * Dc + h * Dc + i, mask=False, other=0.0)
            out_ptr_val += attn_t * val
            tl.store(out_ptr + b * H * Dc + h * Dc + i, out_ptr_val)

    # Store lse[b, h] = lse
    tl.store(lse_ptr + b * H + h, lse)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure device is CUDA for Triton
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA for Triton."

        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Dc = q_nope.shape[2]
        Dp = q_pe.shape[2]

        # Prepare outputs and lse
        output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=q_nope.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        # Make inputs contiguous for Triton
        q_nope_c = q_nope.contiguous()
        q_pe_c = q_pe.contiguous()
        # Kc_all and Kp_all are already [num_tokens, Dc] and [num_tokens, Dp]; we will index via kv_indices
        # We need to squeeze dim=1 for the original ckv_cache/kpe_cache which were [num_pages, 1, D]
        Kc_all = ckv_cache.squeeze(1).contiguous()  # [num_pages, Dc]
        Kp_all = kpe_cache.squeeze(1).contiguous()  # [num_pages, Dp]

        # Loop over batch elements and launch kernel per (b, h)
        for b in range(B):
            # Determine number of tokens for this batch element from kv_indptr
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L_tokens = page_end - page_beg
            if L_tokens <= 0:
                # Output zero and lse = -inf
                output[b].zero_()
                lse[b] = -float('inf')
                continue

            # Gather token indices for this batch
            tok_idx = kv_indices[page_beg:page_end].to(torch.long)

            # Gather Kc and Kp rows for these tokens
            Kc = Kc_all[tok_idx].contiguous()  # [L_tokens, Dc]
            Kp = Kp_all[tok_idx].contiguous()  # [L_tokens, Dp]

            # Launch Triton kernel: one program per (b, h)
            grid = (B, H)
            _batch_elem_kernel[grid](
                q_nope_c, q_pe_c,
                Kc, Kp,
                output, lse,
                B, H, Dc, Dp,
                L_tokens,
                float(sm_scale),
                num_warps=1, num_stages=1
            )

        return output, lse


# Original helper functions
def get_inputs():
    # Example inputs; device='cuda' for Triton execution
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device=device)
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device=device)
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device=device)
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device=device)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), torch.cumsum(_lens, 0)]).to(torch.int32)
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32, device=device)
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


# Original Model for reference
class Model(torch.nn.Module):
    def forward(self, *args):
        return run(*args)