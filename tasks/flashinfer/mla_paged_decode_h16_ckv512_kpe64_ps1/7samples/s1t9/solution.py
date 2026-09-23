import math
import torch
import triton
import triton.language as tl


@triton.jit
def matvec_add_kernel(
    qn_ptr,           # *float32, [D]
    qp_ptr,           # *float32, [Dp]
    Kc_ptr,           # *float32, [L, D], row-major (L, D)
    Kp_ptr,           # *float32, [L, Dp], row-major (L, Dp)
    v_ptr,            # *float32, [L]
    L: tl.int32,      # number of tokens
    D: tl.int32,      # head_dim_ckv
    Dp: tl.int32,     # head_dim_kpe
    BLOCK_K: tl.constexpr,
):
    # One program per output index i in [0, L)
    i = tl.program_id(0)
    sum1 = 0.0
    sum2 = 0.0
    # Reduce over Kc dimension (D)
    for k in range(0, D, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        mask_k = k_off < D
        qn_slice = tl.load(qn_ptr + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
        kc_ptr = Kc_ptr + i * D + k_off
        kc_slice = tl.load(kc_ptr, mask=mask_k, other=0.0)  # [BLOCK_K]
        sum1 += tl.sum(qn_slice * kc_slice, axis=0)
    # Reduce over Kp dimension (Dp)
    for p in range(0, Dp, BLOCK_K):
        p_off = p + tl.arange(0, BLOCK_K)
        mask_p = p_off < Dp
        qp_slice = tl.load(qp_ptr + p_off, mask=mask_p, other=0.0)  # [BLOCK_K]
        kp_ptr = Kp_ptr + i * Dp + p_off
        kp_slice = tl.load(kp_ptr, mask=mask_p, other=0.0)  # [BLOCK_K]
        sum2 += tl.sum(qp_slice * kp_slice, axis=0)
    v = sum1 + sum2
    tl.store(v_ptr + i, v)


@triton.jit
def lse_kernel(
    v_ptr,            # *float32, [L]
    lse_out_ptr,      # *float32, scalar output [1]
    L: tl.int32,
    inv_ln2: tl.float32,
    BLOCK: tl.constexpr,
):
    # One program that scans v to compute max, then sum, then lse
    # Phase 1: compute max
    m = tl.full((), -float('inf'), tl.float32)
    for start in range(0, L, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < L
        v_chunk = tl.load(v_ptr + offs, mask=mask, other=-float('inf'))
        m = tl.maximum(m, tl.max(v_chunk, axis=0))
    # Phase 2: compute sum exp((v - m) * inv_ln2)
    s = 0.0
    for start in range(0, L, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < L
        v_chunk = tl.load(v_ptr + offs, mask=mask, other=-float('inf'))
        contrib = tl.exp((v_chunk - m) * inv_ln2)
        s += tl.sum(contrib, axis=0)
    lse = tl.log(s) + m  # logsumexp over base-2 with inv_ln2
    tl.store(lse_out_ptr, lse)


@triton.jit
def softmax_base2_kernel(
    v_ptr,            # *float32, [L]
    attn_ptr,         # *float32, [L]
    lse_scalar,       # float32 scalar lse value
    L: tl.int32,
    inv_ln2: tl.float32,
    BLOCK: tl.constexpr,
):
    # One program that writes normalized attn: exp((v - lse)/ln(2))
    for start in range(0, L, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < L
        v_chunk = tl.load(v_ptr + offs, mask=mask, other=0.0)
        attn_chunk = tl.exp((v_chunk - lse_scalar) * inv_ln2)
        tl.store(attn_ptr + offs, attn_chunk, mask=mask)


@triton.jit
def matvec_write_y_kernel(
    attn_ptr,         # *float32, [L]
    Kc_ptr,           # *float32, [L, D], row-major (L, D)
    out_ptr,          # *float32, [D]
    L: tl.int32,
    D: tl.int32,
    BLOCK: tl.constexpr,
):
    # One program per output index d in [0, D)
    d = tl.program_id(0)
    sum_acc = 0.0
    # For each token i, accumulate attn[i] * Kc[i, d]
    for i in range(0, L):
        attn_i = tl.load(attn_ptr + i)
        kc_i_d = tl.load(Kc_ptr + i * D + d)
        sum_acc += attn_i * kc_i_d
    tl.store(out_ptr + d, sum_acc)


def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    """
    Triton-only implementation of the original forward.
    - q_nope: [B, 16, 512], bfloat16
    - q_pe: [B, 16, 64], bfloat16
    - ckv_cache: [N, 1, 512], bfloat16
    - kpe_cache: [N, 1, 64], bfloat16
    - kv_indptr: [B+1], int32
    - kv_indices: [M], int32 (M can exceed needed, handled by range)
    - sm_scale: float32 scalar
    Returns: (output [B, 16, 512] bfloat16), (lse [B] float32)
    """
    assert q_nope.dim() == 3 and q_pe.dim() == 3
    assert ckv_cache.dim() == 3 and kpe_cache.dim() == 3
    assert kv_indptr.dim() == 1 and kv_indices.dim() == 1
    assert q_nope.shape[1] == 16 and q_nope.shape[2] == 512
    assert q_pe.shape[1] == 16 and q_pe.shape[2] == 64
    assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1
    assert kv_indptr[-1].item() == kv_indices.numel()  # consistency check

    device = q_nope.device
    B = q_nope.shape[0]
    head_dim_ckv = 512
    head_dim_kpe = 64

    # Prepare Kc_all and Kp_all (view as rows)
    Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [N, 512]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [N, 64]

    output = torch.zeros((B, 16, head_dim_ckv), dtype=torch.float32, device=device)
    lse = torch.full((B, 16), -float('inf'), dtype=torch.float32, device=device)

    inv_ln2 = 1.0 / math.log(2.0)

    # Preallocate attention buffer [L] and output row [D] for each (b, j)
    # We'll re-use buffers inside kernels; here, pass pointers each time.

    for b in range(B):
        # Determine token range
        L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
        if L_tokens <= 0:
            continue

        # Gather token indices and corresponding Kc, Kp for this batch
        tok_idx = (kv_indices[kv_indptr[b]:kv_indptr[b + 1]]).to(torch.int64).contiguous()
        Kc = Kc_all[tok_idx].contiguous()  # [L_tokens, 512], float32
        Kp = Kp_all[tok_idx].contiguous()  # [L_tokens, 64], float32

        # Loop over heads j
        for j in range(16):
            # Compute v[j, :] = sum_i (qn[j,:] · Kc[i,:]) + sum_i (qp[j,:] · Kp[i,:])
            qn_j = q_nope[b, j].to(torch.float32).contiguous()  # [512]
            qp_j = q_pe[b, j].to(torch.float32).contiguous()   # [64]
            v = torch.empty(L_tokens, dtype=torch.float32, device=device)

            # Launch Triton kernel: matvec_add_kernel
            grid_v = (L_tokens,)
            matvec_add_kernel[grid_v](
                qn_j, qp_j, Kc, Kp, v,
                L_tokens, head_dim_ckv, head_dim_kpe,
                BLOCK_K=128, num_warps=4
            )

            # Compute lse_j (base-2 logsumexp) via Triton
            lse_j = torch.empty((), dtype=torch.float32, device=device)
            grid_lse = (1,)
            lse_kernel[grid_lse](
                v, lse_j,
                L_tokens, inv_ln2,
                BLOCK=128, num_warps=2
            )
            lse[b, j] = lse_j

            # Compute attn_j = exp((v - lse_j)/ln(2))
            attn = torch.empty(L_tokens, dtype=torch.float32, device=device)
            grid_softmax = (L_tokens,)
            softmax_base2_kernel[grid_softmax](
                v, attn, lse[b, j], L_tokens, inv_ln2,
                BLOCK=128, num_warps=2
            )

            # Final matvec: out[b, j, :] = attn_j @ Kc[:, :] for each column d
            out_row = torch.empty(head_dim_ckv, dtype=torch.float32, device=device)
            grid_y = (head_dim_ckv,)
            matvec_write_y_kernel[grid_y](
                attn, Kc, out_row,
                L_tokens, head_dim_ckv,
                BLOCK=128, num_warps=4
            )

            output[b, j, :] = out_row

    # Cast output to bfloat16 to match original, keep lse as float32
    output_bf16 = output.to(torch.bfloat16)
    return output_bf16, lse


def get_inputs():
    # Helper to generate inputs for local testing; harness may supply its own inputs.
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    # Construct simple indptr and indices for a single batch
    B = 1
    num_tokens = 8
    kv_indptr = torch.tensor([0, num_tokens], dtype=torch.int32, device='cuda')
    kv_indices = torch.randint(0, 989669, [num_tokens], dtype=torch.int32, device='cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = run(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Run the Triton-orchestrated computation. Ensure CUDA.
        for i in range(len(args)):
            if isinstance(args[i], torch.Tensor) and args[i].device.type != 'cuda':
                args[i] = args[i].to('cuda')
        return run(*args)


def run(*args):
    return ModelNew()(*args)
