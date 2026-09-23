import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_kernel(
    qn_ptr,        # *f32 [H, CK] flattened
    qp_ptr,        # *f32 [H, KP] flattened
    Kc_ptr,        # *f32 [L_tokens, CK]
    Kp_ptr,        # *f32 [L_tokens, KP]
    logits_ptr,    # *f32 [H, L_tokens]
    H: tl.constexpr, CK: tl.constexpr, KP: tl.constexpr, L_tokens: tl.constexpr, sm_scale: tl.constexpr
):
    # program ids: 2D grid over (h, t)
    h = tl.program_id(0)
    t = tl.program_id(1)

    # Reduce over CK for qn[h, :] · Kc[t, :]
    offs_c = tl.arange(0, CK)
    qn_vec = tl.load(qn_ptr + h * CK + offs_c)
    Kc_vec = tl.load(Kc_ptr + t * CK + offs_c)
    dot_qn = tl.sum(qn_vec * Kc_vec, axis=0)

    # Reduce over KP for qp[h, :] · Kp[t, :]
    offs_k = tl.arange(0, KP)
    qp_vec = tl.load(qp_ptr + h * KP + offs_k)
    Kp_vec = tl.load(Kp_ptr + t * KP + offs_k)
    dot_qp = tl.sum(qp_vec * Kp_vec, axis=0)

    logits_ht = sm_scale * (dot_qn + dot_qp)
    tl.store(logits_ptr + h * L_tokens + t, logits_ht)


@triton.jit
def compute_lse_kernel(
    logits_ptr,    # *f32 [H, L_tokens]
    lse_ptr,       # *f32 [H]
    H: tl.constexpr, L_tokens: tl.constexpr
):
    h = tl.program_id(0)
    max_val = -float('inf')
    sum_exp = 0.0

    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        # track max for numerical stability
        max_val = tl.maximum(max_val, val)
        # accumulate exp over tokens
        sum_exp += tl.exp(val - max_val)

    lse = max_val + tl.log(sum_exp)
    tl.store(lse_ptr + h, lse)


@triton.jit
def compute_output_kernel(
    logits_ptr,    # *f32 [H, L_tokens]
    lse_ptr,       # *f32 [H]
    sm_scale,      # f32 scalar
    Kc_ptr,        # *f32 [L_tokens, CK]
    out_ptr,       # *f32 [H, CK]
    H: tl.constexpr, CK: tl.constexpr, L_tokens: tl.constexpr
):
    h = tl.program_id(0)
    # initialize output vector for this head
    offs_c = tl.arange(0, CK)
    out_vec = tl.zeros((CK,), dtype=tl.float32)

    # Loop over tokens, accumulate softmax-prob * Kc[t, :] into out_vec
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        scaled = val * sm_scale
        # softmax probability for this token
        sum_exp = 0.0
        for j in range(0, L_tokens):
            vj = tl.load(logits_ptr + h * L_tokens + j)
            sum_exp += tl.exp(vj * sm_scale)
        prob = tl.exp(scaled) / sum_exp
        Kc_t_vec = tl.load(Kc_ptr + t * CK + offs_c)
        out_vec += prob * Kc_t_vec

    # Store accumulated output for this head
    tl.store(out_ptr + h * CK + offs_c, out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        q_nope: [B, H, CK], bfloat16
        q_pe:   [B, H, KP], bfloat16
        ckv_cache: [num_pages, 1, CK], bfloat16 (squeeze ok)
        kpe_cache: [num_pages, 1, KP], bfloat16
        kv_indptr: [len_indptr], int32 (cumsum of tokens per batch)
        kv_indices: [num_kv_indices], int32 (token indices)
        sm_scale: float32 scalar
        Returns:
        output: [B, H, CK], bfloat16
        lse:    [B, H], float32
        """
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "Tensors must be on CUDA"
        device = q_nope.device

        B, H, CK = q_nope.shape
        _, _, KP = q_pe.shape
        # Squeeze cached caches
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, CK]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, KP]

        # Prepare outputs
        output = torch.empty((B, H, CK), dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Process each batch element
        for b in range(B):
            # Determine number of tokens for this batch via kv_indptr
            if kv_indptr.numel() != len(kv_indptr):
                # Ensure valid
                pass
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())

            if L_tokens <= 0:
                # No tokens, output zeros and arbitrary lse; but to keep consistency, set output zeros and lse -inf
                output[b] = torch.zeros((H, CK), dtype=torch.float32, device=device)
                lse[b] = torch.full((H,), -float('inf'), dtype=torch.float32, device=device)
                continue

            # Gather corresponding Kc/Kp rows
            tok_idx = kv_indices[0:L_tokens].to(torch.int32)  # ensure correct dtype
            Kc = Kc_all[tok_idx]  # [L_tokens, CK]
            Kp = Kp_all[tok_idx]  # [L_tokens, KP]

            # Prepare inputs for kernels
            qn = q_nope[b].to(torch.float32).contiguous()  # [H, CK] flattened by kernel as H rows
            qp = q_pe[b].to(torch.float32).contiguous()    # [H, KP]

            # Allocate logits buffer [H, L_tokens]
            logits = torch.empty((H, L_tokens), dtype=torch.float32, device=device)

            # Launch kernel 1: compute logits
            grid = (H, L_tokens)
            compute_logits_kernel[grid](
                qn, qp, Kc, Kp, logits,
                H=H, CK=CK, KP=KP, L_tokens=L_tokens, sm_scale=float(sm_scale)
            )

            # Launch kernel 2: compute per-head lse
            grid_lse = (H,)
            compute_lse_kernel[grid_lse](
                logits, lse[b],
                H=H, L_tokens=L_tokens
            )

            # Launch kernel 3: compute output
            compute_output_kernel[(H,)](
                logits, lse[b], float(sm_scale), Kc, output[b],
                H=H, CK=CK, L_tokens=L_tokens
            )

        # Return output in bfloat16, and lse as float32 (tuple)
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
