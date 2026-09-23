import torch
import triton
import triton.language as tl


# Triton kernel: compute Y_diag[b, c, i, h, d] = sum_j L[b, c, i, j, h] * hidden[b, c, j, h, d]
# hidden: [B, C, S, H, D], L: [B, C, S, S, H]
@triton.jit
def y_diag_reduce_kernel(hidden_ptr, L_ptr, out_ptr,
                          B, C, S, H, D,
                          stride_hb, stride_hc, stride_hs, stride_hh, stride_hd,
                          stride_lb, stride_lc, stride_li, stride_lj, stride_lh,
                          stride_ob, stride_oc, stride_oi, stride_oh, stride_od,
                          BLOCK_J: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_d = tl.program_id(4)

    if (pid_b >= B) or (pid_c >= C) or (pid_i >= S) or (pid_h >= H) or (pid_d >= D):
        return

    acc = 0.0

    # Loop over j in chunks of BLOCK_J
    for start in range(0, S, BLOCK_J):
        j = start + tl.arange(0, BLOCK_J)
        mask_j = j < S

        # Load hidden[b, c, j, h, d]
        hidden_idx = (
            pid_b * stride_hb +
            pid_c * stride_hc +
            j * stride_hs +
            pid_h * stride_hh +
            pid_d * stride_hd
        )
        hidden_vals = tl.load(hidden_ptr + hidden_idx, mask=mask_j, other=0.0)

        # Load L[b, c, i, j, h]
        L_idx = (
            pid_b * stride_lb +
            pid_c * stride_lc +
            pid_i * stride_li +
            j * stride_lj +
            pid_h * stride_lh
        )
        L_vals = tl.load(L_ptr + L_idx, mask=mask_j, other=0.0)

        # Multiply and reduce over j in this chunk
        prod = L_vals * hidden_vals
        acc += tl.sum(prod, axis=0)

    # Store output Y[b, c, i, h, d] = acc
    out_idx = (
        pid_b * stride_ob +
        pid_c * stride_oc +
        pid_i * stride_oi +
        pid_h * stride_oh +
        pid_d * stride_od
    )
    tl.store(out_ptr + out_idx, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Compute output analogous to the original run using Triton for the heavy reduction.
        - hidden_states: [B, C, S, H, head_dim]
        - A_cumsum: [B, H, C, S]
        Return Y_diag: [B, C, S, H, head_dim] in bfloat16.
        """
        # Extract shapes
        Bsz, num_chunks, S, num_heads, head_dim = hidden_states.shape

        # Ensure inputs are contiguous and on CUDA
        hidden = hidden_states.contiguous().to(torch.float32)
        A = A_cumsum.contiguous().to(torch.float32)

        # Compute cumsum along S for each (b, h, c): cumsum_A[b, h, c, j] = sum_{t=0..j} A[b, h, c, t]
        cumsum_A = torch.cumsum(A, dim=-1)  # [B, H, C, S]

        # Build L in PyTorch: L[i, j, h] = exp(cumsum_A[b, h, c, j]) if i >= j else 0
        # Materialize L as float32 for numerical stability
        L = torch.empty((Bsz, num_chunks, S, S, num_heads), dtype=torch.float32, device=hidden_states.device)

        # Fill L using broadcasting across i >= j
        for b in range(Bsz):
            for c in range(num_chunks):
                for h in range(num_heads):
                    # cumsum_A[b, h, c, :] is the row vector
                    row = cumsum_A[b, h, c]  # [S]
                    # Broadcast row across i dimension
                    # i, j mesh
                    i = torch.arange(S, device=hidden_states.device).view(S, 1)
                    j = torch.arange(S, device=hidden_states.device).view(1, S)
                    mask = i >= j  # lower-triangular mask
                    L[b, c, :, :, h] = torch.exp(row)  # fill all i rows
                    L[b, c, :, :, h][~mask] = 0.0  # set upper-triangular to 0

        # Allocate output Y in fp32 for accumulation
        Y = torch.empty((Bsz, num_chunks, S, num_heads, head_dim), dtype=torch.float32, device=hidden_states.device)

        # Prepare strides
        stride_hb = hidden.stride(0)
        stride_hc = hidden.stride(1)
        stride_hs = hidden.stride(2)
        stride_hh = hidden.stride(3)
        stride_hd = hidden.stride(4)

        stride_lb = L.stride(0)
        stride_lc = L.stride(1)
        stride_li = L.stride(2)
        stride_lj = L.stride(3)
        stride_lh = L.stride(4)

        stride_ob = Y.stride(0)
        stride_oc = Y.stride(1)
        stride_oi = Y.stride(2)
        stride_oh = Y.stride(3)
        stride_od = Y.stride(4)

        # Launch Triton kernel over (b, c, i, h, d)
        grid = (Bsz, num_chunks, S, num_heads, head_dim)
        # BLOCK_J: cover S=128; set 128 for robustness
        y_diag_reduce_kernel[grid](hidden, L, Y,
                                   Bsz, num_chunks, S, num_heads, head_dim,
                                   stride_hb, stride_hc, stride_hs, stride_hh, stride_hd,
                                   stride_lb, stride_lc, stride_li, stride_lj, stride_lh,
                                   stride_ob, stride_oc, stride_oi, stride_oh, stride_od,
                                   BLOCK_J=128)

        # Cast output to bfloat16 to match expected dtype
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
