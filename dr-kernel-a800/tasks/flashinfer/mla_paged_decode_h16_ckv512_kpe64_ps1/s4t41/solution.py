import torch
import triton
import triton.language as tl


@triton.jit
def compute_logit_scalar_kernel(
    qn_ptr,        # *float32, [Kc_dim=512]
    Kc_row_ptr,    # *float32, [Kc_dim]
    qp_ptr,        # *float32, [Kp_dim=64]
    Kp_row_ptr,    # *float32, [Kp_dim]
    out_ptr,       # *float32, scalar output [1]
    Kc_dim: tl.constexpr,     # 512
    Kp_dim: tl.constexpr       # 64
):
    # Accumulate two dot products and store into out_ptr[0]
    acc1 = 0.0
    for k0 in range(0, Kc_dim, 64):
        offs_k = k0 + tl.arange(0, 64)
        mask_k = offs_k < Kc_dim
        a1 = tl.load(qn_ptr + offs_k, mask=mask_k, other=0.0)
        b1 = tl.load(Kc_row_ptr + offs_k, mask=mask_k, other=0.0)
        acc1 += tl.sum(a1 * b1, axis=0)

    acc2 = 0.0
    for k0 in range(0, Kp_dim, 32):
        offs_k = k0 + tl.arange(0, 32)
        mask_k = offs_k < Kp_dim
        a2 = tl.load(qp_ptr + offs_k, mask=mask_k, other=0.0)
        b2 = tl.load(Kp_row_ptr + offs_k, mask=mask_k, other=0.0)
        acc2 += tl.sum(a2 * b2, axis=0)

    tl.store(out_ptr, acc1 + acc2)


@triton.jit
def lse_softmax_row_kernel(
    logits_ptr,   # *float32, [M]
    M,            # int
    scale,        # float32 scalar
    lse_out_ptr,  # *float32, [1] output scalar per run
    BLOCK_M: tl.constexpr = 256
):
    # Numerically stable logsumexp over M with runtime loop
    maxv = -float("inf")
    # Compute max over logits * scale
    m = 0
    while m < M:
        x = tl.load(logits_ptr + m)
        # scale is applied outside: we feed logits already scaled in caller
        # Here we compute max over raw logits; caller scales logits before launch.
        maxv = tl.maximum(maxv, x)
        m += 1

    sum_exp = 0.0
    m = 0
    while m < M:
        x = tl.load(logits_ptr + m)
        sum_exp += tl.exp(x - maxv)
        m += 1

    lse_val = tl.log(sum_exp) + maxv  # logsumexp of scaled logits (scale already applied before launching)
    # Store per run: we launch once per head; if multiple heads, we'll call separately. For lse per head, use a loop.
    tl.store(lse_out_ptr, lse_val)


@triton.jit
def matvec_out_row_kernel(
    attn_ptr,       # *float32, [M]
    Kc_ptr,         # *float32, [M, Kc_dim]
    out_vec_ptr,    # *float32, [Kc_dim]
    M,              # int
    Kc_dim: tl.constexpr,     # 512
    BLOCK_K: tl.constexpr = 128
):
    # out_vec[k] = sum_i attn[i] * Kc[i, k]
    for k0 in range(0, Kc_dim, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < Kc_dim
        acc = tl.zeros([BLOCK_K], dtype=tl.float32)
        i = 0
        while i < M:
            ai = tl.load(attn_ptr + i)
            bi = tl.load(Kc_ptr + i * Kc_dim + offs_k, mask=mask_k, other=0.0)
            acc += ai * bi
            i += 1
        tl.store(out_vec_ptr + offs_k, acc, mask=mask_k)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be CUDA."

        device = q_nope.device
        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]  # 512
        head_dim_kpe = q_pe.shape[2]    # 64

        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Process each batch b
        for b in range(batch_size):
            # Extract tokens range for this batch
            assert kv_indptr.shape[0] == batch_size + 1, "kv_indptr length must be batch_size + 1."
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            M = end - start
            if M <= 0:
                output[b].zero_()
                lse[b].fill_(-float("inf"))
                continue

            # Gather token indices and corresponding rows from caches
            tok_idx = kv_indices[start:end].to(torch.int64).contiguous()  # [M]
            Kc_all = ckv_cache[:, 0, :].to(torch.float32).contiguous()   # [N, 512]
            Kp_all = kpe_cache[:, 0, :].to(torch.float32).contiguous()   # [N, 64]
            # Extract relevant rows
            Kc = Kc_all[tok_idx]  # [M, 512]
            Kp = Kp_all[tok_idx]  # [M, 64]

            # Compute logits per head and per token
            logits = torch.empty(M, dtype=torch.float32, device=device)  # per head
            for h in range(num_qo_heads):
                qn_h = q_nope[b, h].to(torch.float32).contiguous()       # [512]
                qp_h = q_pe[b, h].to(torch.float32).contiguous()         # [64]
                # For each token i, compute scalar logits[i]
                for i in range(M):
                    Kc_i = Kc[i].contiguous()                            # [512]
                    Kp_i = Kp[i].contiguous()                            # [64]
                    out_i = torch.empty(1, dtype=torch.float32, device=device)
                    compute_logit_scalar_kernel[(1,)](
                        qn_h, Kc_i, qp_h, Kp_i, out_i, Kc_dim=512, Kp_dim=64, num_warps=1
                    )
                    logits[i] = out_i[0]

            # Compute lse per head: logsumexp(logits * sm_scale) / ln(2)
            for h in range(num_qo_heads):
                logits_scaled = logits * float(sm_scale)                 # scale applied
                lse_scalar = torch.empty(1, dtype=torch.float32, device=device)
                lse_softmax_row_kernel[(M,)](
                    logits_scaled, M, 1.0, lse_scalar, BLOCK_M=256, num_warps=1
                )
                lse[b, h] = lse_scalar[0]

            # Compute output per head: out = attn @ Kc
            for h in range(num_qo_heads):
                # attn[i] = exp(logits_scaled[i]) / sum_j exp(logits_scaled[j])
                denom = 0.0
                for i in range(M):
                    denom += torch.exp(logits_scaled[i])
                attn = torch.empty(M, dtype=torch.float32, device=device)
                for i in range(M):
                    attn[i] = torch.exp(logits_scaled[i]) / denom

                out_vec = torch.empty(head_dim_ckv, dtype=torch.float32, device=device)
                matvec_out_row_kernel[(head_dim_ckv,)](
                    attn, Kc, out_vec, M, Kc_dim=512, BLOCK_K=128, num_warps=1
                )
                output[b, h] = out_vec.to(torch.bfloat16)

        return output, lse


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
