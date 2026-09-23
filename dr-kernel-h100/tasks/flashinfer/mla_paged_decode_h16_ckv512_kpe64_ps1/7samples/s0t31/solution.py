import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 1: compute lse = log(sum exp(logits_scaled)) / log(2) for a given head h
# Also computes max(logits_scaled) for numerical stability and writes it to lse_ptr[0] (not used).
@triton.jit
def compute_lse_and_max_kernel(
    qnh_ptr,          # [D] pointer to float32, D=512
    Kc_ptr,           # [L_tokens, D] pointer to float32, contiguous
    Kp_ptr,           # [L_tokens, Kp] pointer to float32, contiguous
    lse_ptr,          # [1] pointer to float32, scalar output
    L_TOKENS: tl.constexpr,  # number of tokens, compile-time for unrolling
    sm_scale: tl.constexpr,  # scalar float
    D: tl.constexpr,         # 512
    Kp_const: tl.constexpr,  # 64
    log2: tl.constexpr       # 1.4426950408889634
):
    # We assume single program per (b, h). Gather qnh and compute max and sum for all tokens.
    # Initialize m and res
    m = -float('inf')
    res = 0.0  # sum of exp(logits_scaled - m * sm_scale)

    # Loop over tokens t
    for t in tl.static_range(L_TOKENS):
        # Compute dot(qnh, Kc[t, :]) and dot(qph, Kp[t, :])
        # Row pointers: Kc_row_ptr = Kc_ptr + t * D
        Kc_row_ptr = Kc_ptr + t * D
        Kp_row_ptr = Kp_ptr + t * Kp_const

        # Load qnh
        # Since qnh_ptr is [D], we can load directly
        qnh = tl.load(qnh_ptr + tl.arange(0, D))
        # Load Kc row and Kp row
        Kc_row = tl.load(Kc_row_ptr + tl.arange(0, D))
        Kp_row = tl.load(Kp_row_ptr + tl.arange(0, Kp_const))

        # Compute logits
        dot_qnKc = tl.sum(qnh * Kc_row, axis=0)
        dot_qpKp = tl.sum(qph * Kp_row, axis=0)  # qph is scalar vector [Kp_const]
        logits = dot_qnKc + dot_qpKp  # scalar
        scaled = logits * sm_scale
        m_new = tl.maximum(m, scaled)
        # res accumulates exp(scaled - m_new * sm_scale)
        res += tl.exp(scaled - m_new * sm_scale)
        m = m_new

    # lse = log(res) / log(2)
    lse_val = tl.log(res) * (1.0 / log2)
    # write scalar
    tl.store(lse_ptr, lse_val)


# Kernel 2: compute output[b, h, :] using precomputed lse[b, h]
@triton.jit
def compute_output_with_attn_kernel(
    qnh_ptr,          # [D] pointer to float32, D=512
    Kc_ptr,           # [L_tokens, D] pointer to float32, contiguous
    Kp_ptr,           # [L_TOKENS, Kp] pointer to float32, contiguous
    out_vec_ptr,      # [D] pointer to float32, output vector
    lse_val,          # scalar float32: lse[b, h]
    L_TOKENS: tl.constexpr,  # number of tokens
    sm_scale: tl.constexpr,  # scalar float
    D: tl.constexpr,         # 512
    Kp_const: tl.constexpr   # 64
):
    for t in tl.static_range(L_TOKENS):
        Kc_row_ptr = Kc_ptr + t * D
        Kp_row_ptr = Kp_ptr + t * Kp_const

        qnh = tl.load(qnh_ptr + tl.arange(0, D))
        Kc_row = tl.load(Kc_row_ptr + tl.arange(0, D))
        Kp_row = tl.load(Kp_row_ptr + tl.arange(0, Kp_const))

        dot_qnKc = tl.sum(qnh * Kc_row, axis=0)
        dot_qpKp = tl.sum(qph * Kp_row, axis=0)  # qph is scalar vector [Kp_const]
        logits = dot_qnKc + dot_qpKp
        scaled = logits * sm_scale
        attn = tl.exp(scaled - lse_val)  # softmax probability at token t
        out_vec = tl.load(out_vec_ptr + tl.arange(0, D))  # vector accumulator
        out_vec += attn * tl.load(Kc_row_ptr + tl.arange(0, D))  # attn * Kc[t, :]
        tl.store(out_vec_ptr + tl.arange(0, D), out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA device for Triton
        device = q_nope.device
        if not q_nope.is_cuda:
            q_nope = q_nope.to('cuda')
        if not q_pe.is_cuda:
            q_pe = q_pe.to('cuda')
        if not ckv_cache.is_cuda:
            ckv_cache = ckv_cache.to('cuda')
        if not kpe_cache.is_cuda:
            kpe_cache = kpe_cache.to('cuda')
        if not kv_indptr.is_cuda:
            kv_indptr = kv_indptr.to('cuda')
        if not kv_indices.is_cuda:
            kv_indices = kv_indices.to('cuda')

        B, H, D = q_nope.shape
        assert q_pe.shape == (B, H, 64), "q_pe must have shape [B, 16, 64]"
        # Compute output and lse as float32, then cast output to bfloat16 at the end
        output = torch.zeros((B, H, D), dtype=torch.float32, device=device)
        lse = torch.full((B, H), -float('inf'), dtype=torch.float32, device=device)

        # Prepare qph vector for Kp dot (same shape as qpe, but we pass qnh separately)
        # We will not use qph in these kernels since we pass qnh, but we need qph for Kp rows. Instead,
        # compute qph outside and pass a dummy vector; however, since Triton kernel here does not use qph,
        # we can keep qph as a tensor and use it in Python. For Triton, we pass qnh only.

        # For each batch and head, compute L_tokens and run kernels if L_tokens > 0
        for b in range(B):
            # Compute L_tokens
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            # If no tokens, skip (avoid out-of-bounds)
            if L_tokens <= 0:
                # lse stays -inf, output stays zero
                continue

            # Gather selected tokens
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            tok_idx = kv_indices[start:end].to(torch.long)  # indices in [0, num_pages)

            # Gather Kc_selected and Kp_selected
            Kc_selected = ckv_cache[tok_idx].contiguous().to(torch.float32)  # [L_tokens, D]
            Kp_selected = kpe_cache[tok_idx].contiguous().to(torch.float32)  # [L_tokens, Kp_const]
            D_const = D
            Kp_const = 64
            log2 = 1.4426950408889634

            # qnh: [D], qph: [Kp_const]
            qnh = q_nope[b].contiguous().to(torch.float32)  # [D]
            # qph is not used in Triton kernels (we compute Kp dot in Python), so we can skip passing it.
            # Instead, we compute everything in Triton using qnh only. To compute qph*Kp, we can do it in Python:
            # Compute logits_scaled vector per head using Triton for max and sum, then compute output in Triton.
            # We need qph vector. We can construct it from q_pe:
            qph = q_pe[b].contiguous().to(torch.float32)  # [Kp_const]

            # Compute lse for head h=0..15: we launch one program per (b, h). In this code, we have H heads.
            for h in range(H):
                # Launch Triton kernel to compute lse and max (only lse is returned; max is not stored)
                lse_buf = torch.empty(1, dtype=torch.float32, device=device)
                compute_lse_and_max_kernel[(1,)](
                    qnh, Kc_selected, Kp_selected, lse_buf,  # pointers
                    L_TOKENS=L_tokens,
                    sm_scale=sm_scale,
                    D=D_const,
                    Kp_const=Kp_const,
                    log2=log2
                )
                lse_val = float(lse_buf.item())  # scalar
                # For numerical consistency, store lse per (b, h)
                lse[b, h] = lse_val

                # Compute output[b, h, :]
                out_vec = torch.zeros(D, dtype=torch.float32, device=device)
                compute_output_with_attn_kernel[(1,)](
                    qnh, Kc_selected, Kp_selected, out_vec, lse_val,
                    L_TOKENS=L_tokens,
                    sm_scale=sm_scale,
                    D=D_const,
                    Kp_const=Kp_const
                )
                output[b, h, :] = out_vec

        # Cast output to bfloat16 as in original interface
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def get_inputs():
    # Example inputs; evaluator may override axes. Ensure CUDA device for Triton.
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    num_pages = 989669
    ckv_cache = torch.randn([num_pages, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([num_pages, 1, 64], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to('cuda')
    kv_indices = torch.randint(0, num_pages, [8], dtype=torch.int32).to('cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = ModelNew()(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
