import torch
import triton
import triton.language as tl


@triton.jit
def build_L_kernel(
    A_ptr,            # *ptr to A_cumsum: [B, H, C, S]
    L_ptr,            # *ptr to L: [B, C, S, S, H]
    Bsz,              # int
    C,                # int
    S: tl.constexpr,  # 128
    H: tl.constexpr,  # 32
    stride_ab, stride_ah, stride_ac, stride_as,  # strides for A
    stride_lb, stride_lc, stride_li, stride_lj, stride_lh  # strides for L
):
    # Grid: (B, C, H)
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)

    # Compute cumsum along j for A[b, h, c, :]
    for j in range(0, S):
        a_val = tl.load(A_ptr + b * stride_ab + h * stride_ah + c * stride_ac + j * stride_as)
        a_val = a_val.to(tl.float32)
        # cumsum_j[j] = sum_{t=0..j} A[b,h,c,t]
        cumsum_j = tl.zeros((), dtype=tl.float32)
        for t in range(0, j + 1):
            # load A[b,h,c,t] and accumulate
            v = tl.load(A_ptr + b * stride_ab + h * stride_ah + c * stride_ac + t * stride_as)
            cumsum_j += v.to(tl.float32)
        exp_val = tl.exp(cumsum_j)
        # Store to L[i, j, h] for all i >= j
        for i in range(0, S):
            if i >= j:
                tl.store(L_ptr + b * stride_lb + c * stride_lc + i * stride_li + j * stride_lj + h * stride_lh, exp_val)


@triton.jit
def compute_Y_diag_kernel(
    L_ptr,            # *ptr to L: [B, C, S, S, H]
    hidden_ptr,       # *ptr to hidden_states: [B, C, S, H, head_dim]
    out_ptr,          # *ptr to output Y: [B, C, S, H, head_dim]
    Bsz,              # int
    C,                # int
    S: tl.constexpr,  # 128
    H: tl.constexpr,  # 32
    head_dim,         # int
    stride_lb, stride_lc, stride_li, stride_lj, stride_lh,  # strides for L
    stride_hb, stride_hc, stride_hs, stride_hh, stride_hd,  # strides for hidden
    stride_ob, stride_oc, stride_oi, stride_oh, stride_od   # strides for output
):
    # Grid: (B, C, S, H, head_dim)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    d = tl.program_id(4)

    acc = tl.zeros((), dtype=tl.float32)
    # Reduce over j: Y[i] = sum_j L[i, j, h] * hidden[b, c, j, h, d]
    for j in range(0, S):
        l_val = tl.load(L_ptr + b * stride_lb + c * stride_lc + i * stride_li + j * stride_lj + h * stride_lh)
        h_val = tl.load(hidden_ptr + b * stride_hb + c * stride_hc + j * stride_hs + h * stride_hh + d * stride_hd)
        acc += l_val.to(tl.float32) * h_val.to(tl.float32)

    tl.store(out_ptr + b * stride_ob + c * stride_oc + i * stride_oi + h * stride_oh + d * stride_od, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Shapes:
        # hidden_states: [B, C, S, H, head_dim]
        # A_cumsum: [B, H, C, S]
        Bsz, C, S, H, head_dim = hidden_states.shape

        # Allocate L: [B, C, S, S, H] in float32
        L = torch.empty((Bsz, C, S, S, H), dtype=torch.float32, device=hidden_states.device)

        # Ensure contiguous tensors for predictable strides
        hidden_states_c = hidden_states.contiguous()
        A_cumsum_c = A_cumsum.contiguous()
        L_c = L.contiguous()

        # Launch Triton kernel to build L
        grid1 = (Bsz, C, H)
        build_L_kernel[grid1](
            A_cumsum_c, L_c,
            Bsz, C, S=128, H=32,
            stride_ab=A_cumsum_c.stride(0), stride_ah=A_cumsum_c.stride(1), stride_ac=A_cumsum_c.stride(2), stride_as=A_cumsum_c.stride(3),
            stride_lb=L_c.stride(0), stride_lc=L_c.stride(1), stride_li=L_c.stride(2), stride_lj=L_c.stride(3), stride_lh=L_c.stride(4)
        )

        # Allocate output: [B, C, S, H, head_dim] in float32
        out = torch.empty((Bsz, C, S, H, head_dim), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernel to compute Y_diag
        grid2 = (Bsz, C, S, H, head_dim)
        compute_Y_diag_kernel[grid2](
            L_c, hidden_states_c, out,
            Bsz, C, S=128, H=32, head_dim=head_dim,
            stride_lb=L_c.stride(0), stride_lc=L_c.stride(1), stride_li=L_c.stride(2), stride_lj=L_c.stride(3), stride_lh=L_c.stride(4),
            stride_hb=hidden_states_c.stride(0), stride_hc=hidden_states_c.stride(1), stride_hs=hidden_states_c.stride(2), stride_hh=hidden_states_c.stride(3), stride_hd=hidden_states_c.stride(4),
            stride_ob=out.stride(0), stride_oc=out.stride(1), stride_oi=out.stride(2), stride_oh=out.stride(3), stride_od=out.stride(4)
        )

        # Cast to bfloat16 to match the original output dtype
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
