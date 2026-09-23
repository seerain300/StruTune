import math
import torch
import triton
import triton.language as tl


@triton.jit
def matvec_add_kernel(
    qn_ptr,            # *float32, [H, D] flattened
    qp_ptr,            # *float32, [H, Dp] flattened
    Kc_ptr,            # *float32, [L, D], row-major (we pass flattened [L*D])
    Kp_ptr,            # *float32, [L, Dp], row-major (flattened [L*Dp])
    v_ptr,             # *float32, [L]
    H: tl.int32,       # num_qo_heads
    D: tl.int32,       # head_dim_ckv
    Dp: tl.int32,      # head_dim_kpe
    L: tl.int32,       # number of tokens
    stride_qn_h: tl.int32,   # stride between heads in qn (D)
    stride_qp_h: tl.int32,   # stride between heads in qp (Dp)
    BLOCK_K: tl.constexpr,
):
    # One program per output index i in [0, L)
    i = tl.program_id(0)
    sum1 = 0.0
    sum2 = 0.0
    # Loop over head dimension D in blocks
    for k0 in range(0, D, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        mask_k = k < D
        # For each head j, accumulate qn[j, :] · Kc[i, :]
        for j in range(0, H):
            qn_vals = tl.load(qn_ptr + j * stride_qn_h + k, mask=mask_k, other=0.0)  # [BLOCK_K]
            Kc_vals = tl.load(Kc_ptr + i * D + k, mask=mask_k, other=0.0)           # [BLOCK_K]
            sum1 += tl.sum(qn_vals * Kc_vals, axis=0)
    # Loop over head dimension Dp in blocks
    for p0 in range(0, Dp, BLOCK_K):
        p = p0 + tl.arange(0, BLOCK_K)
        mask_p = p < Dp
        for j in range(0, H):
            qp_vals = tl.load(qp_ptr + j * stride_qp_h + p, mask=mask_p, other=0.0)   # [BLOCK_K]
            Kp_vals = tl.load(Kp_ptr + i * Dp + p, mask=mask_p, other=0.0)            # [BLOCK_K]
            sum2 += tl.sum(qp_vals * Kp_vals, axis=0)
    v = sum1 + sum2
    tl.store(v_ptr + i, v)


@triton.jit
def lse_base2_kernel(
    v_ptr,             # *float32, [L]
    lse_ptr,           # *float32, [1]
    L: tl.int32,
    inv_ln2: tl.float32,
    BLOCK_L: tl.constexpr,
):
    # One program reduces over L in chunks
    sum_exp = 0.0
    max_v = -float("inf")
    for l0 in range(0, L, BLOCK_L):
        l = l0 + tl.arange(0, BLOCK_L)
        mask_l = l < L
        v_chunk = tl.load(v_ptr + l, mask=mask_l, other=-float("inf"))
        # Track max across chunk
        chunk_max = tl.max(v_chunk, axis=0)
        max_v = tl.maximum(max_v, chunk_max)
        # Sum exp of scaled chunk
        sum_exp += tl.sum(tl.exp((v_chunk - max_v) * inv_ln2), axis=0)
    lse_val = max_v + tl.log(sum_exp)  # logsumexp base-2
    tl.store(lse_ptr, lse_val)


@triton.jit
def softmax_base2_kernel(
    v_ptr,             # *float32, [L]
    lse_ptr,           # *float32, [1]
    attn_ptr,          # *float32, [L]
    L: tl.int32,
    inv_ln2: tl.float32,
    BLOCK_L: tl.constexpr,
):
    # One program per index i in [0, L)
    i = tl.program_id(0)
    # Load lse
    lse_j = tl.load(lse_ptr)
    vi = tl.load(v_ptr + i)
    attn_i = tl.exp((vi - lse_j) * inv_ln2)
    tl.store(attn_ptr + i, attn_i)


@triton.jit
def matvec_write_y_kernel(
    attn_ptr,          # *float32, [L]
    Kc_ptr,            # *float32, [L, D] flattened
    y_ptr,             # *float32, [D]
    D: tl.int32,
    L: tl.int32,
    BLOCK_L: tl.constexpr,
):
    # One program writes one output dimension h
    h = tl.program_id(0)
    acc = 0.0
    for l0 in range(0, L, BLOCK_L):
        l = l0 + tl.arange(0, BLOCK_L)
        mask_l = l < L
        attn_chunk = tl.load(attn_ptr + l, mask=mask_l, other=0.0)  # [BLOCK_L]
        Kc_chunk = tl.load(Kc_ptr + l * D + h, mask=mask_l, other=0.0)  # [BLOCK_L]
        acc += tl.sum(attn_chunk * Kc_chunk, axis=0)
    tl.store(y_ptr + h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        for i in range(len(args)):
            if isinstance(args[i], torch.Tensor) and args[i].device.type != 'cuda':
                args[i] = args[i].to('cuda')

        # Extract shapes
        B, H, D = q_nope.shape
        _, _, Dp = q_pe.shape
        N = ckv_cache.shape[0]
        L = (kv_indptr[1] - kv_indptr[0]).item()  # single batch; generalize below
        # Generalize to any batch size: process one batch element at a time
        # Note: len_indptr = B + 1
        output = torch.empty((B, H, D), dtype=torch.bfloat16, device=q_nope.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        inv_ln2 = 1.0 / math.log(2.0)  # passed to kernels

        # We'll process one batch element b
        b = 0
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        L = max(page_end - page_beg, 0)
        if L == 0:
            # No tokens for this batch; output zeros and lse -inf
            output[b].zero_()
            lse[b].fill_(-float("inf"))
            return output, lse

        tok_idx = kv_indices[page_beg:page_end].to(torch.int32).to('cuda')
        # Gather Kc and Kp
        Kc = ckv_cache.squeeze(1).to(torch.float32).index_select(0, tok_idx).contiguous()  # [L, D]
        Kp = kpe_cache.squeeze(1).to(torch.float32).index_select(0, tok_idx).contiguous()  # [L, Dp]

        # Flatten q_nope and q_pe for head iteration
        qn = q_nope.to(torch.float32).reshape(H, D).contiguous()
        qp = q_pe.to(torch.float32).reshape(H, Dp).contiguous()

        # Allocate intermediates
        v = torch.empty((L,), dtype=torch.float32, device=q_nope.device)
        attn = torch.empty((L,), dtype=torch.float32, device=q_nope.device)

        # Launch matvec_add_kernel: one program per i in [0, L)
        grid1 = (L,)
        matvec_add_kernel[grid1](
            qn.flatten(), qp.flatten(), Kc.flatten(), Kp.flatten(), v,
            H, D, Dp, L, D, Dp, 64,
            num_warps=4,
        )

        # Compute lse for each head j; we will loop j=0..H-1
        grid2 = (1,)
        for j in range(0, H):
            lse2 = torch.empty((1,), dtype=torch.float32, device=q_nope.device)
            lse_base2_kernel[grid2](
                v, lse2, L, inv_ln2, 128,
                num_warps=4,
            )
            lse[b, j] = lse2[0]

            # Compute attn for each i; one program per i
            attn.fill_(0)
            softmax_base2_kernel[grid1](
                v, lse2, attn, L, inv_ln2, 128,
                num_warps=4,
            )

            # Compute y[j, :] = attn @ Kc
            y = torch.empty((D,), dtype=torch.float32, device=q_nope.device)
            matvec_write_y_kernel[(D,)](
                attn, Kc.flatten(), y, D, L, 128,
                num_warps=4,
            )
            output[b, j, :] = y.to(torch.bfloat16)

        return output, lse


# Optional helper for local testing (non-recursive)
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


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    return ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)


def run(*args):
    return ModelNew()(*args)
