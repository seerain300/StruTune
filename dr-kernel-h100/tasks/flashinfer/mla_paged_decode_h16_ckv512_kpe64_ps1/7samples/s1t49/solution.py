import math
import torch
import triton
import triton.language as tl


@triton.jit
def matvec_add_kernel(
    qn_ptr,            # *float32, flattened over [H, D]
    qp_ptr,            # *float32, flattened over [H, Dp]
    Kc_ptr,            # *float32, flattened over [L, D]
    Kp_ptr,            # *float32, flattened over [L, Dp]
    v_ptr,             # *float32, [L]
    H: tl.int32,       # num_qo_heads
    D: tl.int32,       # head_dim_ckv
    Dp: tl.int32,      # head_dim_kpe
    L: tl.int32,       # number of tokens per batch
    stride_qn_h: tl.int32,   # stride between heads in qn (D)
    stride_qp_h: tl.int32,   # stride between heads in qp (Dp)
    BLOCK_K: tl.constexpr,
):
    # One program per output index i in [0, L)
    i = tl.program_id(0)
    sum1 = 0.0
    sum2 = 0.0
    # Loop over Kc dimension D in blocks
    for k0 in range(0, D, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        mask_k = k < D
        # Accumulate qn[j, :] · Kc[i, :]
        for j in range(0, H):
            qn_vals = tl.load(qn_ptr + j * stride_qn_h + k, mask=mask_k, other=0.0)  # [BLOCK_K]
            Kc_vals = tl.load(Kc_ptr + i * D + k, mask=mask_k, other=0.0)           # [BLOCK_K]
            sum1 += tl.sum(qn_vals * Kc_vals, axis=0)
    # Loop over Kp dimension Dp in blocks
    for p0 in range(0, Dp, BLOCK_K):
        p = p0 + tl.arange(0, BLOCK_K)
        mask_p = p < Dp
        for j in range(0, H):
            qp_vals = tl.load(qp_ptr + j * stride_qp_h + p, mask=mask_p, other=0.0)  # [BLOCK_K]
            Kp_vals = tl.load(Kp_ptr + i * Dp + p, mask=mask_p, other=0.0)           # [BLOCK_K]
            sum2 += tl.sum(qp_vals * Kp_vals, axis=0)
    # Write v[i] = sum1 + sum2
    tl.store(v_ptr + i, sum1 + sum2)


@triton.jit
def lse_base2_kernel(
    v_ptr,             # *float32, [L]
    lse_ptr,           # *float32, [H]
    H: tl.int32,
    L: tl.int32,
    inv_ln2: tl.float32,
    BLOCK_L: tl.constexpr,
):
    # One program per head j
    j = tl.program_id(0)
    # Track max over v
    v_max = -float('inf')
    for l0 in range(0, L, BLOCK_L):
        l = l0 + tl.arange(0, BLOCK_L)
        mask_l = l < L
        v_seg = tl.load(v_ptr + l, mask=mask_l, other=-float('inf'))
        v_max = tl.maximum(v_max, tl.max(v_seg, axis=0))
    # Sum of exp((v - v_max) * inv_ln2)
    sum_exp = 0.0
    for l0 in range(0, L, BLOCK_L):
        l = l0 + tl.arange(0, BLOCK_L)
        mask_l = l < L
        v_seg = tl.load(v_ptr + l, mask=mask_l, other=0.0)
        # scaled = (v - v_max) * inv_ln2
        scaled = (v_seg - v_max) * inv_ln2
        sum_exp += tl.sum(tl.exp(scaled), axis=0)
    # lse = v_max + ln(2) * log(sum_exp)
    lse = v_max + tl.log(sum_exp) / inv_ln2
    tl.store(lse_ptr + j, lse)


@triton.jit
def softmax_base2_kernel(
    v_ptr,             # *float32, [L]
    lse_ptr,           # *float32, [H]
    attn_ptr,          # *float32, [L]
    H: tl.int32,
    L: tl.int32,
    inv_ln2: tl.float32,
    BLOCK_L: tl.constexpr,
):
    # One program per head j; write attn for all i
    j = tl.program_id(0)
    lse_j = tl.load(lse_ptr + j)
    for l0 in range(0, L, BLOCK_L):
        l = l0 + tl.arange(0, BLOCK_L)
        mask_l = l < L
        v_seg = tl.load(v_ptr + l, mask=mask_l, other=0.0)
        scaled = (v_seg - lse_j) * inv_ln2
        attn_seg = tl.exp(scaled)
        tl.store(attn_ptr + l, attn_seg, mask=mask_l)


@triton.jit
def matvec_write_y_kernel(
    attn_ptr,          # *float32, [L]
    Kc_ptr,            # *float32, flattened over [L, D] (we pass contiguous [L, D] as flattened)
    y_ptr,             # *float32, [D]
    D: tl.int32,
    L: tl.int32,
    BLOCK_L: tl.constexpr,
):
    # One program per output dimension h
    h = tl.program_id(0)
    acc = 0.0
    for l0 in range(0, L, BLOCK_L):
        l = l0 + tl.arange(0, BLOCK_L)
        mask_l = l < L
        attn_seg = tl.load(attn_ptr + l, mask=mask_l, other=0.0)  # [BLOCK_L]
        Kc_seg = tl.load(Kc_ptr + l * D + h, mask=mask_l, other=0.0)  # [BLOCK_L]
        acc += tl.sum(attn_seg * Kc_seg, axis=0)
    tl.store(y_ptr + h, acc)


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    # Ensure inputs are on CUDA
    device = q_nope.device
    B = q_nope.shape[0]
    H = q_nope.shape[1]
    D = q_nope.shape[2]
    Dp = q_pe.shape[2]
    N = ckv_cache.shape[0]

    # Cast to float32 for Triton math
    qn_flat = q_nope.to(torch.float32).reshape(H, D)          # [H, D], flattened in kernel
    qp_flat = q_pe.to(torch.float32).reshape(H, Dp)           # [H, Dp]
    Kc = ckv_cache.squeeze(1).to(torch.float32)               # [N, D]
    Kp = kpe_cache.squeeze(1).to(torch.float32)               # [N, Dp]

    # Output tensors
    output = torch.empty((B, H, D), dtype=torch.float32, device=device)
    lse = torch.empty((B, H), dtype=torch.float32, device=device)

    # Process each batch
    inv_ln2 = 1.0 / math.log(2.0)

    for b in range(B):
        # Compute token range for this batch
        start = int(kv_indptr[b].item())
        end = int(kv_indptr[b + 1].item())
        if end <= start:
            # No tokens for this batch
            output[b] = torch.zeros((H, D), dtype=torch.float32, device=device)
            lse[b] = torch.full((H,), -float('inf'), dtype=torch.float32, device=device)
            continue

        L = end - start
        tok_idx = kv_indices[start:end].to(torch.int32)

        Kc_sel = Kc[tok_idx]  # [L, D], contiguous
        Kp_sel = Kp[tok_idx]  # [L, Dp], contiguous

        # Launch matvec-add kernels: compute v per head
        v = torch.empty((L,), dtype=torch.float32, device=device)
        # One program per i
        grid_v = (L,)
        matvec_add_kernel[grid_v](
            qn_flat, qp_flat, Kc_sel.reshape(-1), Kp_sel.reshape(-1), v,
            H, D, Dp, L, D, Dp,
            BLOCK_K=64,
        )

        # Scale v for logsumexp
        v_scaled = v * sm_scale

        # Compute lse per head
        grid_lse = (H,)
        lse_kernel = lse_base2_kernel
        lse_kernel[grid_lse](
            v_scaled, lse[b], H, L, inv_ln2,
            BLOCK_L=128,
        )

        # Compute attention per head
        attn = torch.empty((L,), dtype=torch.float32, device=device)
        softmax_base2_kernel[(H,)](
            v_scaled, lse[b], attn, H, L, inv_ln2,
            BLOCK_L=128,
        )

        # Compute final output y for each head h
        for j in range(H):
            y_row = torch.empty((D,), dtype=torch.float32, device=device)
            grid_y = (1,)
            matvec_write_y_kernel[grid_y](
                attn, Kc_sel.reshape(-1), y_row, D, L,
                BLOCK_L=128,
            )
            output[b, j, :] = y_row

    # Cast output to bfloat16 to match original
    output = output.to(torch.bfloat16)
    return output, lse


# Helper for local testing (not used by evaluation harness)
def get_inputs():
    batch_size = 1
    num_qo_heads = 16
    head_dim_ckv = 512
    head_dim_kpe = 64
    num_pages = 989669

    device = 'cuda'
    q_nope = torch.randn([batch_size, num_qo_heads, head_dim_ckv], dtype=torch.bfloat16, device=device)
    q_pe = torch.randn([batch_size, num_qo_heads, head_dim_kpe], dtype=torch.bfloat16, device=device)
    ckv_cache = torch.randn([num_pages, 1, head_dim_ckv], dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn([num_pages, 1, head_dim_kpe], dtype=torch.bfloat16, device=device)
    # Simple indptr for a single batch
    kv_indptr = torch.tensor([0, 10], dtype=torch.int32, device=device)  # length = 2
    # Token indices
    kv_indices = torch.randint(0, num_pages, [10], dtype=torch.int32, device=device)
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Ensure inputs are on CUDA for Triton
        for i in range(len(args)):
            if isinstance(args[i], torch.Tensor) and args[i].device.type != 'cuda':
                args[i] = args[i].to('cuda')
        return run(*args)


def run(*args):
    return ModelNew()(*args)
