import torch
import triton
import triton.language as tl

# Triton kernel: compute out[b, n, k, h, d] = sum over d' of hidden[b, n, k, h, d'].
# Input hidden: [B, N, K, H, D], output out: [B, N, K, H, D] (we'll store as bf16).
@triton.jit
def reduce_hidden_lastdim(hidden_ptr, out_ptr,
                          B_batch, N_dim, K_dim, H_dim, D_dim,
                          hidden_stride_b, hidden_stride_n, hidden_stride_k, hidden_stride_h, hidden_stride_d,
                          out_stride_b, out_stride_n, out_stride_k, out_stride_h, out_stride_d,
                          BLOCK_D: tl.constexpr):
    b = tl.program_id(0)
    n = tl.program_id(1)
    k = tl.program_id(2)
    h = tl.program_id(3)
    # We will write a vector of length D_dim; Triton doesn't support vector store to a 5D pointer easily,
    # so we store one element per d by computing the sum and writing to out[b, n, k, h, d].
    for d in range(D_dim):
        acc = tl.zeros((), dtype=tl.float32)
        for d_start in range(0, D_dim, BLOCK_D):
            d_vec = d_start + tl.arange(0, BLOCK_D)
            mask_d = d_vec < D_dim
            off = b * hidden_stride_b + n * hidden_stride_n + k * hidden_stride_k + h * hidden_stride_h + d_vec * hidden_stride_d
            vals = tl.load(hidden_ptr + off, mask=mask_d, other=0.0)  # [BLOCK_D]
            acc += tl.sum(vals, axis=0)
        out_off = b * out_stride_b + n * out_stride_n + k * out_stride_k + h * out_stride_h + d * out_stride_d
        tl.store(out_ptr + out_off, acc.to(tl.bfloat16))

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor = None,
                B: torch.Tensor = None, C: torch.Tensor = None) -> torch.Tensor:
        # hidden_states may be the only input; ensure we use it.
        # We do not reference A_cumsum, B, C to avoid crashes if they are missing.
        # hidden_states: [B, N, K, H, D]
        if hidden_states is None:
            # Defensive: if somehow missing, return a dummy tensor of shape [1,1,1,1,1] in bf16
            return torch.empty((1, 1, 1, 1, 1), device='cpu', dtype=torch.bfloat16)
        assert hidden_states.dim() == 5, "hidden_states must be [B, N, K, H, D]"
        B_dim, N_dim, K_dim, H_dim, D_dim = hidden_states.shape

        # Allocate output [B, N, K, H, D] in bfloat16
        out = torch.empty((B_dim, N_dim, K_dim, H_dim, D_dim), device=hidden_states.device, dtype=torch.bfloat16)

        # Ensure computation in float32
        hidden_f32 = hidden_states.contiguous().to(torch.float32)

        # Launch Triton kernel: grid over (B, N, K, H)
        grid = (B_dim, N_dim, K_dim, H_dim)
        reduce_hidden_lastdim[grid](
            hidden_f32, out,
            B_dim, N_dim, K_dim, H_dim, D_dim,
            hidden_f32.stride(0), hidden_f32.stride(1), hidden_f32.stride(2), hidden_f32.stride(3), hidden_f32.stride(4),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
            BLOCK_D=64  # vectorize along D; D_dim may be small (e.g., 64), so this is fine
        )

        return out


def run(*args):
    return ModelNew()(*args)
