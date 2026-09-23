import torch
import math
import triton
import triton.language as tl


@triton.jit
def _compute_lse_kernel(
    qn_ptr,          # *float32, [D]
    qp_ptr,          # *float32, [DP]
    Kc_ptr,          # *float32, [num_pages, D]
    Kp_ptr,          # *float32, [num_pages, DP]
    tok_idx_ptr,     # *int32,   [L_TOKENS]
    lse_ptr,         # *float32, [num_qo_heads]
    D: tl.constexpr,          # 512
    DP: tl.constexpr,         # 64
    L_TOKENS: tl.constexpr,   # number of selected tokens
    scale: tl.constexpr,      # sm_scale (float32)
):
    # Each program handles one head (grid over num_qo_heads)
    h = tl.program_id(0)
    # Pointers for q vectors of this head
    qn = tl.load(qn_ptr + h * D)            # [D]
    qp = tl.load(qp_ptr + h * DP)           # [DP]

    # Vector of token indices
    t = tl.arange(0, L_TOKENS)               # [L_TOKENS]
    idx = tl.load(tok_idx_ptr + t)           # [L_TOKENS], int32

    # Gather K rows for all tokens at once (vectorized)
    Kc_rows = tl.load(Kc_ptr + idx * D)      # [L_TOKENS, D], float32
    Kp_rows = tl.load(Kp_ptr + idx * DP)     # [L_TOKENS, DP], float32

    # Compute logits per token: dot(qn, Kc_row) + dot(qp, Kp_row)
    # Broadcast q vectors to [1, D] / [1, DP] and sum along the feature axis
    logits_vec = tl.sum(Kc_rows * qn[None, :], axis=1) + tl.sum(Kp_rows * qp[None, :], axis=1)  # [L_TOKENS]
    logits_scaled = logits_vec * scale

    # Compute logsumexp base-2: lse = m + log(sum(exp(logits_scaled - m))) / log(2)
    m = tl.max(logits_scaled, axis=0)                           # scalar
    s = tl.sum(tl.exp(logits_scaled - m), axis=0)               # scalar
    ln2 = 0.6931471805599453
    lse_scalar = m + tl.log(s) * scale / ln2                    # scalar

    # Store lse per head
    tl.store(lse_ptr + h, lse_scalar)


@triton.jit
def _compute_attention_output_kernel(
    qn_ptr,          # *float32, [D]
    qp_ptr,          # *float32, [DP]
    Kc_ptr,          # *float32, [num_pages, D]
    Kp_ptr,          # *float32, [num_pages, DP]
    tok_idx_ptr,     # *int32,   [L_TOKENS]
    lse_ptr,         # *float32, [num_qo_heads]
    out_ptr,         # *float32, [num_qo_heads, D]
    D: tl.constexpr,          # 512
    DP: tl.constexpr,         # 64
    L_TOKENS: tl.constexpr,   # number of selected tokens
    scale: tl.constexpr,      # sm_scale (float32)
):
    # Each program handles one head (grid over num_qo_heads)
    h = tl.program_id(0)

    # Load q vectors
    qn = tl.load(qn_ptr + h * D)           # [D]
    qp = tl.load(qp_ptr + h * DP)          # [DP]

    # Token indices and gathered rows
    t = tl.arange(0, L_TOKENS)
    idx = tl.load(tok_idx_ptr + t)          # [L_TOKENS], int32
    Kc_rows = tl.load(Kc_ptr + idx * D)     # [L_TOKENS, D]
    Kp_rows = tl.load(Kp_ptr + idx * DP)    # [L_TOKENS, DP]

    # Compute logits and scaled logits
    logits_vec = tl.sum(Kc_rows * qn[None, :], axis=1) + tl.sum(Kp_rows * qp[None, :], axis=1)  # [L_TOKENS]
    logits_scaled = logits_vec * scale

    # Load lse for this head
    lse_scalar = tl.load(lse_ptr + h)  # scalar float32
    # Compute attention weights
    exp_scale = tl.exp((logits_scaled - lse_scalar) * scale)  # [L_TOKENS]
    sum_exp = tl.sum(exp_scale, axis=0)                        # scalar
    attn_vec = exp_scale / sum_exp                            # [L_TOKENS]

    # Output vector: sum_t attn[t] * Kc_rows[t, :]
    # Broadcast attn_vec[:, None] to [L_TOKENS, D] and reduce over tokens
    out_vec = tl.sum(attn_vec[:, None] * Kc_rows, axis=0)     # [D]
    tl.store(out_ptr + h * D, out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Device and dtype setup
        device = q_nope.device
        B, H, D = q_nope.shape
        _, _, DP = q_pe.shape
        assert H == 16, "num_qo_heads must be 16"
        assert D == 512, "head_dim_ckv must be 512"
        assert DP == 64, "head_dim_kpe must be 64"

        # Prepare gathered caches (Kc_all, Kp_all) as float32
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, D]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, DP]

        # Output and lse tensors
        output = torch.empty((B, H, D), dtype=torch.float32, device=device)  # per-head vectors [D]
        lse = torch.empty((B, H), dtype=torch.float32, device=device)         # per-head scalar

        # Process each batch
        for b in range(B):
            # Determine number of tokens for this batch
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No tokens in this batch for this b: output zeros
                output[b].zero_()
                lse[b].zero_()
                continue

            # Gather token indices for this batch
            tok_idx = kv_indices[int(kv_indptr[b].item()): int(kv_indptr[b + 1].item())].to(torch.int32).to(device)

            # Launch kernel to compute lse per head
            grid = (H,)
            _compute_lse_kernel[grid](
                q_nope[b], q_pe[b], Kc_all, Kp_all, tok_idx,
                lse[b],
                D=D, DP=DP, L_TOKENS=L_tokens, scale=float(sm_scale)
            )

            # Launch kernel to compute attention output per head
            _compute_attention_output_kernel[grid](
                q_nope[b], q_pe[b], Kc_all, Kp_all, tok_idx,
                lse[b], output[b],
                D=D, DP=DP, L_TOKENS=L_tokens, scale=float(sm_scale)
            )

        # Cast output to bfloat16 to match original behavior
        output = output.to(torch.bfloat16)
        return output, lse


# Helper to match original get_inputs behavior (not used by evaluator, but provided for completeness)
def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to('cuda')
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32).to('cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


# Fused operator interface expected by evaluator
def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    out = ModelNew()(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return out if isinstance(out, (tuple, list)) else [out]