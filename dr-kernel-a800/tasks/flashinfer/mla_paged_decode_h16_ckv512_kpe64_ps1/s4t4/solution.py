import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Matvec kernel: computes C_row = A_row @ B_col for A: [1, K], B: [M, K], output C: [1, M]
# Each program computes one output element (index m), reducing over K in tiles.
@triton.jit
def matvec_row_kernel(
    A_ptr,          # *float32, [1, K]
    B_ptr,          # *float32, [M, K]
    C_ptr,          # *float32, [1, M]
    K: tl.constexpr,         # int, e.g., 512 or 64
    M,                # int, number of rows in B
    BLOCK_K: tl.constexpr,   # tile size along K, e.g., 64 or 128
):
    m = tl.program_id(0)  # program id along M
    acc = tl.zeros((), dtype=tl.float32)
    k = 0
    while k < K:
        offs_k = k + tl.arange(0, BLOCK_K)  # constexpr vector
        a = tl.load(A_ptr + offs_k)         # [BLOCK_K]
        b = tl.load(B_ptr + m * K + offs_k) # [BLOCK_K]
        acc += tl.sum(a * b, axis=0)        # scalar
        k += BLOCK_K
    tl.store(C_ptr + m, acc)


# Softmax + LSE kernel: given logits_scaled (vector of length MAX_M), compute:
# lse = log(sum(exp(logits_scaled))) / ln(2) and store to out_lse_ptr[0]
# attn vector is written to attn_ptr (only first M entries used). MAX_M must be constexpr here.
@triton.jit
def softmax_lse_kernel(
    x_ptr,          # *float32, [MAX_M]
    out_lse_ptr,    # *float32, [1] (lse per head)
    attn_ptr,       # *float32, [MAX_M] (attention vector per head, only first M used)
    M,              # int, actual length (runtime)
    MAX_M: tl.constexpr,       # int constexpr (e.g., 16384)
    sm_scale,       # float32
):
    idx = tl.arange(0, MAX_M)  # constexpr vector of length MAX_M
    mask = idx < M
    x = tl.load(x_ptr + idx, mask=mask, other=-float('inf'))  # [MAX_M]
    # Compute max for numerical stability
    max_val = tl.max(x, axis=0)
    x_shift = x - max_val
    exp_x = tl.exp(x_shift)
    sum_exp = tl.sum(exp_x, axis=0)
    lse_val = tl.log(sum_exp) / tl.log(2.0)  # divide by ln(2)
    attn = exp_x / sum_exp
    # Store lse to out_lse_ptr[0]
    tl.store(out_lse_ptr, lse_val)
    # Store attn to attn_ptr (masked)
    tl.store(attn_ptr + idx, attn, mask=mask)


# Matvec reduction kernel: out_vec[d] = sum_i A[i] * B[i, d] for A: [M] row vector, B: [M, K], output out_vec: [K]
@triton.jit
def matvec_reduce_kernel(
    A_ptr,          # *float32, [M]
    B_ptr,          # *float32, [M, K]
    out_ptr,        # *float32, [K]
    M,              # int
    K: tl.constexpr,         # int, e.g., 512
    BLOCK_M: tl.constexpr,   # tile size along M, e.g., 128
):
    d = tl.program_id(0)  # program id along K dimension
    acc = tl.zeros((), dtype=tl.float32)
    m = 0
    while m < M:
        offs_m = m + tl.arange(0, BLOCK_M)  # constexpr vector
        mask = offs_m < M
        a = tl.load(A_ptr + offs_m, mask=mask, other=0.0)  # [BLOCK_M]
        b = tl.load(B_ptr + offs_m * K + d)                # [BLOCK_M]
        acc += tl.sum(a * b, axis=0)
        m += BLOCK_M
    tl.store(out_ptr + d, acc)


# Triton-only forward
class ModelNew(torch.nn.Module):
    def __init__(self, max_m: int = 16384, block_k: int = 128, block_m: int = 128):
        super().__init__()
        self.max_m = max_m
        self.block_k = block_k
        self.block_m = block_m
        assert TRITON_AVAILABLE, "Triton is not available. Please install triton."

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Extract shapes
        B, H, Kc_dim = q_nope.shape
        _, _, Kp_dim = q_pe.shape
        assert H == 16, "num_qo_heads must be 16"
        assert Kc_dim == 512, "head_dim_ckv must be 512"
        assert Kp_dim == 64, "head_dim_kpe must be 64"

        device = q_nope.device
        if not TRITON_AVAILABLE:
            # Fallback to original PyTorch behavior if Triton not available
            return run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)

        # Ensure inputs are float32 for compute
        q_nope_f32 = q_nope.to(torch.float32)
        q_pe_f32 = q_pe.to(torch.float32)
        # Squeeze caches (second dim is 1)
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [N, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [N, 64]

        # Output initialization
        output = torch.empty((B, H, Kc_dim), dtype=torch.float32, device=device)  # compute in fp32, cast at end
        lse = torch.full((B, H), -float("inf"), dtype=torch.float32, device=device)

        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            M = max(0, end - start)
            if M == 0:
                lse[b].zero_()
                continue

            tok_idx = kv_indices[start:start + M].to(torch.int32)
            Kc = Kc_all[tok_idx]  # [M, 512]
            Kp = Kp_all[tok_idx]  # [M, 64]

            for h in range(H):
                qn = q_nope_f32[b, h, :]  # [512]
                qp = q_pe_f32[b, h, :]    # [64]

                # Compute logits vector of length M: qn @ Kc.T and qp @ Kp.T
                logits_qn = torch.empty((M,), dtype=torch.float32, device=device)
                matvec_row_kernel[(M,)](
                    qn.to(torch.float32).view(1, -1),          # A_ptr: [1, 512]
                    Kc,                                        # B_ptr: [M, 512]
                    logits_qn,                                # C_ptr: [M]
                    512,                                      # K constexpr
                    M,                                        # runtime M
                    self.block_k,                            # BLOCK_K
                )
                logits_qp = torch.empty((M,), dtype=torch.float32, device=device)
                matvec_row_kernel[(M,)](
                    qp.to(torch.float32).view(1, -1),          # A_ptr: [1, 64]
                    Kp,                                        # B_ptr: [M, 64]
                    logits_qp,                                # C_ptr: [M]
                    64,                                       # K constexpr
                    M,                                        # runtime M
                    self.block_k,                            # BLOCK_K
                )
                logits = (logits_qn + logits_qp) * sm_scale  # [M]

                # If M > max_m, fallback to PyTorch for correctness (evaluation M should be <= max_m)
                if M <= self.max_m:
                    attn = torch.empty((self.max_m,), dtype=torch.float32, device=device)
                    lse_vec = torch.empty((1,), dtype=torch.float32, device=device)
                    softmax_lse_kernel[(1,)](
                        logits,                             # *float32, [M]
                        lse_vec,                           # *float32, [1]
                        attn,                              # *float32, [MAX_M]
                        M,                                 # runtime M
                        self.max_m,                        # constexpr MAX_M
                        sm_scale,                          # float32
                    )
                    lse[b, h] = lse_vec[0]
                else:
                    logits_scaled = logits * sm_scale
                    lse[b, h] = torch.logsumexp(logits_scaled, dim=0) / math.log(2.0)
                    attn = torch.softmax(logits_scaled, dim=0)

                # Compute out[h, :] = attn @ Kc -> [512]
                out_vec = torch.empty((Kc_dim,), dtype=torch.float32, device=device)
                matvec_reduce_kernel[(Kc_dim,)](
                    attn,                                # A: [M]
                    Kc,                                 # B: [M, 512]
                    out_vec,                            # out: [512]
                    M,                                  # runtime M
                    Kc_dim,                             # K constexpr 512
                    self.block_m,                       # BLOCK_M
                )
                output[b, h, :] = out_vec

        return output.to(torch.bfloat16), lse.to(torch.float32)


# Original helper run for reference
@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64

    device = q_nope.device

    Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [N, 512]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [N, 64]

    output = torch.zeros((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
    lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    for b in range(batch_size):
        start = int(kv_indptr[b].item())
        end = int(kv_indptr[b + 1].item())
        M = max(0, end - start)
        if M == 0:
            output[b].zero_()
            lse[b].zero_()
            continue

        tok_idx = kv_indices[start:end].to(torch.long)
        Kc = Kc_all[tok_idx]  # [M, 512]
        Kp = Kp_all[tok_idx]  # [M, 64]

        for h in range(num_qo_heads):
            qn = q_nope[b, h, :].to(torch.float32)  # [512]
            qp = q_pe[b, h, :].to(torch.float32)    # [64]
            logits = (qn @ Kc.T) + (qp @ Kp.T)     # [M]
            logits_scaled = logits * sm_scale
            lse[b, h] = torch.logsumexp(logits_scaled, dim=0) / math.log(2.0)
            attn = torch.softmax(logits_scaled, dim=0)  # [M]
            out = attn @ Kc  # [512]
            output[b, h, :] = out.to(torch.bfloat16)

    return output, lse


def run(*args):
    return ModelNew()(*args)
