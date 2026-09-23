import torch
import triton
import triton.language as tl


@triton.jit
def build_L_kernel(
    A_ptr,            # *float32, shape [B, H, C, S]
    L_ptr,            # *float32, shape [B, C, S, S, H]
    Bsz, Csz, Hsz, S, head_dim,
    stride_A_b, stride_A_h, stride_A_c, stride_A_s,  # strides for A
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,  # strides for L
):
    # program id maps to (b, c, h)
    pid = tl.program_id(axis=0)
    # compute indices
    bc_h_per_line = Csz * Hsz
    b = pid // bc_h_per_line
    rem = pid % bc_h_per_line
    c = rem // Hsz
    h = rem % Hsz

    # base pointers
    A_base = A_ptr + b * stride_A_b + h * stride_A_h + c * stride_A_c
    L_base_b = L_ptr + b * stride_L_b + c * stride_L_c
    L_base_h = L_base_b + h * stride_L_h

    # compute cumsum along s
    cumsum = tl.zeros((), dtype=tl.float32)
    # loop over j from 0 to S-1
    for j in range(0, 128):
        a_val = tl.load(A_base + j * stride_A_s)
        cumsum += a_val
        # set L[i, j, h] for all i >= j
        for i in range(0, 128):
            if i >= j:
                tl.store(L_base_h + i * stride_L_i + j * stride_L_j, tl.exp(cumsum))


@triton.jit
def compute_Y_diag_kernel(
    L_ptr,            # *float32, shape [B, C, S, S, H]
    hidden_ptr,       # *float32, shape [B, C, S, H, head_dim]
    Y_ptr,            # *float32, shape [B, C, S, H, head_dim]
    Bsz, Csz, S, Hsz, head_dim,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
    stride_h_b, stride_h_c, stride_h_s, stride_h_h, stride_h_d,
    stride_Y_b, stride_Y_c, stride_Y_i, stride_Y_h, stride_Y_d,
):
    # program id maps to (b, c, i, h, d)
    pid = tl.program_id(axis=0)
    bc_per_line = Csz * Hsz
    bc = pid // (S * head_dim)
    rem = pid % (S * head_dim)
    i = rem // head_dim
    d = rem % head_dim
    b = bc // Csz
    c = bc % Csz
    h = 0  # h is fixed by pid, we derive from rem but it's not needed since h is part of pid?
    # No, h is already in pid decomposition above: we recompute with bc and rem:
    # We need to reconstruct h from pid. Let's do it properly:
    bc_total = Csz * Hsz
    b = pid // bc_total
    rem2 = pid % bc_total
    c = rem2 // Hsz
    h = rem2 % Hsz

    # base pointers
    L_base = L_ptr + b * stride_L_b + c * stride_L_c
    L_base_h = L_base + h * stride_L_h
    hidden_base = hidden_ptr + b * stride_h_b + c * stride_h_c
    hidden_base_h = hidden_base + h * stride_h_h
    Y_base = Y_ptr + b * stride_Y_b + c * stride_Y_c
    Y_base_h = Y_base + h * stride_Y_h

    # accumulation
    y_val = tl.zeros((), dtype=tl.float32)
    for j in range(0, 128):
        L_val = tl.load(L_base_h + i * stride_L_i + j * stride_L_j)
        hidden_val = tl.load(hidden_base_h + j * stride_h_s + d * stride_h_d)
        y_val += L_val * hidden_val
    # store result
    tl.store(Y_base_h + i * stride_Y_i + d * stride_Y_d, y_val)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Extract shapes
        Bsz, Csz, S, Hsz, head_dim = hidden_states.shape
        # A_cumsum: [B, H, C, S]
        assert A_cumsum.shape == (Bsz, Hsz, Csz, S), "A_cumsum shape must be [B, H, C, S]"
        # Allocate L as float32 (compute buffer)
        L = torch.empty((Bsz, Csz, S, S, Hsz), dtype=torch.float32, device=hidden_states.device)
        # Ensure contiguity (optional, but we pass strides anyway)
        # Launch kernel 1: build L
        grid_L = (Bsz * Csz * Hsz,)
        build_L_kernel[grid_L](
            A_cumsum, L,
            Bsz, Csz, Hsz, S, head_dim,
            A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), A_cumsum.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            num_warps=4, num_stages=2
        )

        # Allocate output Y as float32 (compute buffer)
        Y = torch.empty((Bsz, Csz, S, Hsz, head_dim), dtype=torch.float32, device=hidden_states.device)

        # Launch kernel 2: compute Y_diag
        grid_Y = (Bsz * Csz * S * Hsz * head_dim,)
        compute_Y_diag_kernel[grid_Y](
            L, hidden_states, Y,
            Bsz, Csz, S, Hsz, head_dim,
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2), hidden_states.stride(3), hidden_states.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
            num_warps=4, num_stages=2
        )

        # Return in bfloat16 to match expected output
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
