import torch
import torch.nn as nn

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton LayerNorm forward kernel:
# Input: x_ptr as a 1D contiguous array of length M*D, weight_ptr [D], bias_ptr [D]
# Output: out_ptr [M*D] (flattened view of [M, D])
# Each program handles one row. It computes mean and variance across D in FP32,
# then normalizes and applies affine (gamma, beta).
@triton.jit
def layernorm_fwd_kernel(
    x_ptr,            # *f32, input flattened [M*D]
    w_ptr,            # *f32, gamma [D]
    b_ptr,            # *f32, beta  [D]
    out_ptr,          # *f32, output flattened [M*D]
    M: tl.int32,      # number of rows
    D: tl.int32,      # feature dimension
    eps: tl.float32,  # epsilon
):
    row = tl.program_id(0)
    if row >= M:
        return

    # First pass: compute sum and sum of squares across D
    sum_val = 0.0
    sum_sq = 0.0
    for j in range(0, D):
        x_ij = tl.load(x_ptr + row * D + j)  # load as f32
        sum_val += x_ij
        sum_sq += x_ij * x_ij

    mean = sum_val / D
    var = sum_sq / D - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine, store
    for j in range(0, D):
        x_ij = tl.load(x_ptr + row * D + j)
        norm = (x_ij - mean) * rstd
        gamma_j = tl.load(w_ptr + j)
        beta_j = tl.load(b_ptr + j)
        y_ij = norm * gamma_j + beta_j
        tl.store(out_ptr + row * D + j, y_ij)


class ModelNew(nn.Module):
    def forward(self, *args):
        # Accept a single tuple produced by get_inputs. Extract hidden_states and norm params.
        # We need hidden_states [B, S, D], norm1_weight [D], norm1_bias [D],
        # and norm2_weight [D], norm2_bias [D].
        hidden_state = None
        for t in args:
            if isinstance(t, torch.Tensor):
                hidden_state = t
                break
        if hidden_state is None:
            raise RuntimeError("ModelNew.forward did not receive any tensor input")

        # Determine shape: hidden_state is expected to be [B, S, D]
        if hidden_state.dim() != 3:
            raise RuntimeError("ModelNew.forward expects hidden_states as 3D tensor [B, S, D]")

        B, S, D = hidden_state.shape
        M = B * S

        # Flatten input to [M*D] for the kernel. Ensure contiguous and FP32.
        x_flat = hidden_state.contiguous().view(M, D).to(torch.float32).flatten()

        # Prepare weight and bias for LayerNorms. Infer from args.
        gamma1 = None
        beta1 = None
        gamma2 = None
        beta2 = None

        for t in args:
            # If tensor is 1D and matches D, assume it's a gamma/beta
            if isinstance(t, torch.Tensor) and t.dim() == 1 and t.numel() == D:
                if gamma1 is None:
                    gamma1 = t.to(torch.float32).contiguous()
                elif beta1 is None:
                    beta1 = t.to(torch.float32).contiguous()
                # Try to find a second pair
                if gamma2 is None:
                    gamma2 = t.to(torch.float32).contiguous()
                elif beta2 is None:
                    beta2 = t.to(torch.float32).contiguous()
            # If gamma2 or beta2 not found, reuse gamma1/beta1 for LN2
        if gamma2 is None:
            gamma2 = gamma1
        if beta2 is None:
            beta2 = beta1

        # Output buffers
        y1_flat = torch.empty_like(x_flat)  # [M*D]
        y2_flat = torch.empty_like(x_flat)  # [M*D]

        # Launch LN1 kernel
        grid_ln1 = (M,)
        layernorm_fwd_kernel[grid_ln1](
            x_flat,
            gamma1,
            beta1,
            y1_flat,
            M, D,
            eps=1e-5,
            num_warps=4,
        )

        # Launch LN2 kernel (second LayerNorm)
        grid_ln2 = (M,)
        layernorm_fwd_kernel[grid_ln2](
            y1_flat,
            gamma2,
            beta2,
            y2_flat,
            M, D,
            eps=1e-5,
            num_warps=4,
        )

        # Reshape back to [B, S, D]
        output = y2_flat.view(B, S, D)
        return output


def run(*args):
    return ModelNew()(*args)
