import torch
import triton
import triton.language as tl

# Triton elementwise kernels
@triton.jit
def softplus_torch_like(x_ptr, out_ptr, N):
    # x_ptr: [N], out_ptr: [N]
    offs = tl.arange(0, 1024)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    absx = tl.abs(x)
    soft = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-absx))
    tl.store(out_ptr + offs, soft, mask=mask)

@triton.jit
def sigmoid_torch_like(x_ptr, out_ptr, N):
    # Sigmoid: 1 / (1 + exp(-x))
    offs = tl.arange(0, 1024)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offs, sig, mask=mask)

@triton.jit
def exp_vec(x_ptr, out_ptr, N):
    # Elementwise exp on vector
    offs = tl.arange(0, 1024)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.exp(x)
    tl.store(out_ptr + offs, y, mask=mask)

# Triton GEMV: out[V] = scale * q_vec[K] @ state[K,V], state row-major [K,V]
@triton.jit
def gemv_kernel(q_ptr, state_ptr, out_ptr, K, V, scale, BLOCK_V: tl.constexpr, BLOCK_K: tl.constexpr):
    # One program computes the entire out vector by tiling over V and K.
    # We'll use a single program and iterate over V and K tiles.
    # out_ptr is [V]; we compute out[v] for each v.
    for v_tile in range(0, tl.cdiv(V, BLOCK_V)):
        v_offs = v_tile * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_v = v_offs < V
        acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
        for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
            k_offs = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
            mask_k = k_offs < K
            # Load q[k] and state[v, k] for this tile
            qk = tl.load(q_ptr + k_offs, mask=mask_k, other=0.0)  # [BLOCK_K]
            state_block = tl.load(
                state_ptr + v_offs[:, None] * K + k_offs[None, :],
                mask=mask_v[:, None] & mask_k[None, :],
                other=0.0
            )  # [BLOCK_V, BLOCK_K]
            prod = qk[None, :] * state_block  # [BLOCK_V, BLOCK_K]
            acc += tl.sum(prod, axis=1)  # sum over K for each v
        acc = acc * scale
        tl.store(out_ptr + v_offs, acc, mask=mask_v)

# Triton kernel to update state per (seq_idx, t, h):
# - state_old: [V, K] (row-major), float32
# - k_vec: [K] (k[t, h_src, :])
# - v_vec: [V] (v[t, h, :])
# - beta: scalar (beta[t, h])
# - g: scalar (g[t, h])
# - state_new: [V, K] (float32), output updated in-place
@triton.jit
def update_state_kernel(
    state_old_ptr, k_ptr, v_ptr, beta, g, state_new_ptr, V, K,
    BLOCK_V: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Each program handles one (seq_idx, t_idx, h_idx) and updates state_new for that (h).
    # Load k_vec
    k_vec = tl.load(k_ptr + tl.arange(0, BLOCK_K), mask=tl.arange(0, BLOCK_K) < K, other=0.0)  # [BLOCK_K]
    # Compute old_v = k_vec @ state_old
    old_v = tl.zeros((V,), dtype=tl.float32)
    for tile_v in range(0, tl.cdiv(V, BLOCK_V)):
        v_offs = tile_v * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_v = v_offs < V
        state_block = tl.load(
            state_old_ptr + v_offs[:, None] * K + tl.arange(0, BLOCK_K)[None, :],
            mask=mask_v[:, None],
            other=0.0
        )  # [BLOCK_V, BLOCK_K]
        old_v_tile = tl.sum(state_block * k_vec[None, :], axis=1)  # [BLOCK_V]
        old_v[v_offs] = old_v_tile

    # Compute new_v = beta * v_vec + (1 - beta) * old_v
    v_vec = tl.load(v_ptr + tl.arange(0, V), mask=tl.arange(0, V) < V, other=0.0)  # [V]
    new_v = beta * v_vec + (1.0 - beta) * old_v  # elementwise

    # Compute alpha = sum_k k_vec[k] * old_v[k], beta_new = sum_k k_vec[k] * new_v[k]
    alpha = 0.0
    beta_new = 0.0
    for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = k_offs < K
        k_vec_k = tl.load(k_ptr + k_offs, mask=mask_k, other=0.0)
        old_v_k = old_v[k_offs]
        new_v_k = new_v[k_offs]
        alpha += tl.sum(k_vec_k * old_v_k, axis=0)
        beta_new += tl.sum(k_vec_k * new_v_k, axis=0)

    # Update state_new = g * state_old + (beta_new - alpha)
    # Read state_old block and write state_new block
    for tile_v in range(0, tl.cdiv(V, BLOCK_V)):
        v_offs = tile_v * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_v = v_offs < V
        for tile_k in range(0, tl.cdiv(K, BLOCK_K)):
            k_offs = tile_k * BLOCK_K + tl.arange(0, BLOCK_K)
            mask_k = k_offs < K
            state_old_block = tl.load(
                state_old_ptr + v_offs[:, None] * K + k_offs[None, :],
                mask=mask_v[:, None] & mask_k[None, :],
                other=0.0
            )
            state_new_block = (g * state_old_block) + (beta_new - alpha)
            tl.store(
                state_new_ptr + v_offs[:, None] * K + k_offs[None, :],
                state_new_block,
                mask=mask_v[:, None] & mask_k[None, :]
            )

# ModelNew forward: Triton-only computation
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # Ensure CUDA tensors
        device = q.device
        L = q.shape[0]
        H = 8
        V = 128
        K = 128
        S = cu_seqlens.numel() - 1

        # Move inputs to CUDA if needed
        if not q.is_cuda:
            q = q.cuda(non_blocking=True)
        if not k.is_cuda:
            k = k.cuda(non_blocking=True)
        if not v.is_cuda:
            v = v.cuda(non_blocking=True)
        if not state.is_cuda:
            state = state.cuda(non_blocking=True)
        if not A_log.is_cuda:
            A_log = A_log.cuda(non_blocking=True)
        if not a.is_cuda:
            a = a.cuda(non_blocking=True)
        if not dt_bias.is_cuda:
            dt_bias = dt_bias.cuda(non_blocking=True)
        if not b.is_cuda:
            b = b.cuda(non_blocking=True)
        if not cu_seqlens.is_cuda:
            cu_seqlens = cu_seqlens.cuda(non_blocking=True)

        # Compute g and beta via Triton elementwise kernels
        # a_expanded is [L*8] with elements a[t, h]


def run(*args):
    return ModelNew()(*args)
