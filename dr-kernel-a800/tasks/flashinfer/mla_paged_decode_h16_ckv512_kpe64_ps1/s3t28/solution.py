import torch
import math
import triton
import triton.language as tl


# ----------------------------
# Triton kernels (defined once)
# ----------------------------

@triton.jit
def fused_logits_kernel(
    qn_ptr,      # *f32 [H, Dq]
    qp_ptr,      # *f32 [H, Dp]
    Kc_ptr,      # *f32 [T, Dq]
    Kp_ptr,      # *f32 [T, Dp]
    logits_ptr,  # *f32 [H, T]
    H: tl.constexpr,    # number of heads (16)
    T: tl.constexpr,    # number of tokens (runtime const for this launch)
    Dq: tl.constexpr,   # 512
    Dp: tl.constexpr,   # 64
    BLOCK_T: tl.constexpr,  # e.g., 128 (power of two)
):
    h = tl.program_id(0)  # one program per head
    offs_h = tl.arange(0, 1)  # always 1 since h is scalar
    # Preload qn[h, :] and qp[h, :]
    qn_vec = tl.load(qn_ptr + h * Dq + tl.arange(0, Dq), mask=tl.arange(0, Dq) < Dq, other=0.0)  # [Dq]
    qp_vec = tl.load(qp_ptr + h * Dp + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0)  # [Dp]

    # Iterate over tokens in tiles
    for t_start in range(0, T, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T

        # Kc_sub: [BLOCK_T, Dq]
        Kc_sub = tl.load(Kc_ptr + offs_t[:, None] * Dq + tl.arange(0, Dq)[None, :], mask=mask_t[:, None], other=0.0)
        # Kp_sub: [BLOCK_T, Dp]
        Kp_sub = tl.load(Kp_ptr + offs_t[:, None] * Dp + tl.arange(0, Dp)[None, :], mask=mask_t[:, None], other=0.0)

        # dot(qn[h, :], Kc_sub) -> [BLOCK_T]
        dot_qn = tl.sum(qn_vec[None, :] * Kc_sub, axis=1)
        # dot(qp[h, :], Kp_sub) -> [BLOCK_T]
        dot_qp = tl.sum(qp_vec[None, :] * Kp_sub, axis=1)

        # logits[h, t] for this tile
        logits_vec = dot_qn + dot_qp
        # Store to [H, T] row-major
        tl.store(logits_ptr + h * T + offs_t, logits_vec, mask=mask_t)


@triton.jit
def softmax_row_kernel(
    logits_ptr,   # *f32 [H, T]
    attn_ptr,     # *f32 [H, T]
    H: tl.constexpr,    # 16
    T: tl.constexpr,    # number of tokens
    BLOCK_T: tl.constexpr,  # 128 (power of two)
):
    h = tl.program_id(0)
    # Load row logits[h, :]
    offs = tl.arange(0, BLOCK_T)
    mask = offs < T
    row = tl.load(logits_ptr + h * T + offs, mask=mask, other=-float('inf'))
    # Max for numerical stability
    row_max = tl.max(row, axis=0)
    row_shift = row - row_max
    exp_row = tl.exp(row_shift)
    row_sum = tl.sum(exp_row, axis=0)
    attn_row = exp_row / row_sum
    tl.store(attn_ptr + h * T + offs, attn_row, mask=mask)


@triton.jit
def lse_row_kernel(
    logits_ptr,     # *f32 [H, T]
    lse_ptr,        # *f32 [H]
    H: tl.constexpr,    # 16
    T: tl.constexpr,    # number of tokens
    BLOCK_T: tl.constexpr,  # 128 (power of two)
    sm_scale: tl.constexpr, # float scalar
):
    h = tl.program_id(0)
    offs = tl.arange(0, BLOCK_T)
    mask = offs < T
    row = tl.load(logits_ptr + h * T + offs, mask=mask, other=-float('inf'))
    # Scale
    row_scaled = row * sm_scale
    # logsumexp over row
    row_max = tl.max(row_scaled, axis=0)
    row_shift = row_scaled - row_max
    exp_row = tl.exp(row_shift)
    row_sum = tl.sum(exp_row, axis=0)
    lse_val = tl.log(row_sum) + row_max  # in natural log
    ln2 = 0.6931471805599453  # math.log(2.0)
    lse_val = lse_val / ln2
    tl.store(lse_ptr + h, lse_val)


@triton.jit
def matmul_row_kernel(
    attn_ptr,       # *f32 [H, T]
    Kc_ptr,         # *f32 [T, Dq]
    out_ptr,        # *f32 [H, Dq]
    H: tl.constexpr,    # 16
    T: tl.constexpr,    # number of tokens
    Dq: tl.constexpr,   # 512
    BLOCK_D: tl.constexpr,  # 128 (power of two)
):
    h = tl.program_id(0)
    # out_vec: accumulate across tokens
    out_vec = tl.zeros((Dq,), dtype=tl.float32)
    offs_d = tl.arange(0, BLOCK_D)
    for t in range(0, T):
        attn_val = tl.load(attn_ptr + h * T + t)  # scalar
        Kc_vec = tl.load(Kc_ptr + t * Dq + offs_d, mask=offs_d < Dq, other=0.0)  # [BLOCK_D]
        out_vec += attn_val * Kc_vec
    tl.store(out_ptr + h * Dq + offs_d, out_vec, mask=offs_d < Dq)


# ----------------------------
# ModelNew.forward
# ----------------------------

def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    # Ensure CUDA device
    assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA device for Triton kernels"

    B = q_nope.shape[0]
    H = q_nope.shape[1]
    Dq = q_nope.shape[2]
    Dp = q_pe.shape[2]
    # Output allocation
    output = torch.empty((B, H, Dq), dtype=torch.float32, device=q_nope.device)  # we'll store float32 then cast
    lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

    for b in range(B):
        # Compute token range
        L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
        if L_tokens <= 0:
            # No KV cache for this batch element
            output[b].zero_()
            lse[b].fill_(-float('inf'))
            continue

        # Gather Kc and Kp for this batch element
        tok_idx = kv_indices[int(kv_indptr[b].item()): int(kv_indptr[b + 1].item())]  # [L_tokens]
        Kc_b = ckv_cache[tok_idx].squeeze(1).to(torch.float32)  # [L_tokens, 512]
        Kp_b = kpe_cache[tok_idx].squeeze(1).to(torch.float32)  # [L_tokens, 64]

        # qn and qp for this batch
        qn = q_nope[b].to(torch.float32)  # [16, 512]
        qp = q_pe[b].to(torch.float32)    # [16, 64]

        # Allocate intermediate buffers
        logits = torch.empty((H, L_tokens), dtype=torch.float32, device=q_nope.device)  # [H, T]
        attn = torch.empty((H, L_tokens), dtype=torch.float32, device=q_nope.device)    # [H, T]

        # Launch fused_logits_kernel: one program per head
        grid_logits = (H,)
        fused_logits_kernel[grid_logits](
            qn, qp, Kc_b, Kp_b, logits,
            H=H, T=L_tokens, Dq=Dq, Dp=Dp, BLOCK_T=128,
            num_warps=4, num_stages=2
        )

        # Compute lse per head
        grid_lse = (H,)
        lse_row_kernel[grid_lse](
            logits, lse[b],
            H=H, T=L_tokens, BLOCK_T=128, sm_scale=float(sm_scale),
            num_warps=4, num_stages=2
        )

        # Compute attention via softmax (Triton kernel)
        grid_softmax = (H,)
        softmax_row_kernel[grid_softmax](
            logits, attn,
            H=H, T=L_tokens, BLOCK_T=128,
            num_warps=4, num_stages=2
        )

        # Compute output per head: out[h, :] = attn[h, :] @ Kc_b[:, :]
        out_b = torch.empty((H, Dq), dtype=torch.float32, device=q_nope.device)
        grid_matmul = (H,)
        matmul_row_kernel[grid_matmul](
            attn, Kc_b, out_b,
            H=H, T=L_tokens, Dq=Dq, BLOCK_D=128,
            num_warps=4, num_stages=2
        )

        # Store output for this batch
        output[b] = out_b

    # Cast output to bfloat16 to match original’s output dtype
    output_bf16 = output.to(torch.bfloat16)
    return output_bf16, lse


def get_inputs():
    # Example inputs; real evaluation will provide its own tensors
    device = torch.device('cuda')
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device=device)
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device=device)
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device=device)
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to(device)
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32).to(device)
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return fused_operator(*args)


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    return run(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)


def run(*args):
    return ModelNew()(*args)
