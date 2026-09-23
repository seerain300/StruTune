import torch
import triton
import triton.language as tl


@triton.jit
def softplus_torch_like(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    # softplus(x) = log(1 + exp(x)) in a numerically stable way
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # softplus: for x>0: x + log(1 + exp(-x)); else: log(1 + exp(x))
    pos = x > 0.0
    sp = tl.where(pos, x + tl.log(1.0 + tl.exp(-x)), tl.log(1.0 + tl.exp(x)))
    tl.store(out_ptr + offs, sp, mask=mask)


@triton.jit
def sigmoid_torch_like(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def gemv_kernel(q_ptr, state_ptr, out_ptr, L, H, scale, BLOCK: tl.constexpr):
    # Each program handles a single output (t, h). We loop over t and h inside.
    # q_ptr: [L, H, K], state_ptr: [H, K, V], out_ptr: [L, H, V]
    for t in range(0, L):
        for h in range(0, H):
            q_row_offset = (t * H + h) * 128
            q_vec = tl.load(q_ptr + q_row_offset + tl.arange(0, 128)).to(tl.float32)  # [128]

            # state[h, :, :] flattened as [K*V]
            state_row = state_ptr + h * 128 * 128  # base pointer to start of h-th head
            acc = 0.0
            for kk in range(0, 128, BLOCK):
                k_idx = kk + tl.arange(0, BLOCK)
                mask_k = k_idx < 128
                state_chunk = tl.load(state_row + k_idx * 128, mask=mask_k, other=0.0).to(tl.float32)
                acc += tl.sum(q_vec[kk:kk + BLOCK] * state_chunk, axis=0)

            out_val = acc * scale
            out_off = t * H * 128 + h * 128
            tl.store(out_ptr + out_off, out_val.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        # Ensure CUDA and contiguity
        assert q.is_cuda and k.is_cuda and v.is_cuda and state.is_cuda and A_log.is_cuda and a.is_cuda and b.is_cuda and dt_bias.is_cuda, "All inputs must be CUDA tensors."
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        state = state.contiguous()
        A_log = A_log.contiguous()
        a = a.contiguous()
        b = b.contiguous()

        L, Hq, K = q.shape
        Kk, Hk, Kk2 = k.shape
        Lv, Hv, V = v.shape
        assert Hq == Hk and K == Kk and K == 128 and V == 128 and Hq == 4, "Expected q[k,4,128], k[k,4,128], v[k,8,128]"
        # Expand q/k to 8 heads via repeat_interleave(2)
        H = 8  # expanded heads
        q_exp = q.repeat_interleave(2, dim=1)  # [L, 8, 128]
        k_exp = k.repeat_interleave(2, dim=1)  # [L, 8, 128]

        # Allocate outputs
        output = torch.empty((L, H, V), dtype=torch.bfloat16, device=q.device)
        new_state = torch.empty((cu_seqlens.shape[0] - 1, H, V, V), dtype=torch.float32, device=q.device)

        # Prepare a_expanded and b_expanded: flatten to [L*32]
        a_flat = a.view(-1)                         # [L*32]
        b_flat = b.view(-1)                         # [L*32]
        N1 = a_flat.numel()
        N2 = b_flat.numel()
        N3


def run(*args):
    return ModelNew()(*args)
