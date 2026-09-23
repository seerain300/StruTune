import torch
import triton
import triton.language as tl

# Triton kernel: reduce over the last dimension (D) of a 5D tensor [B, N, K, H, D]
# Produces output of shape [B, N, K, H].
@triton.jit
def reduce_last_dim_kernel(inp_ptr, out_ptr,
                            B, N, K, H, D,
                            inp_stride_b, inp_stride_n, inp_stride_k, inp_stride_h, inp_stride_d,
                            out_stride_b, out_stride_n, out_stride_k, out_stride_h,
                            BLOCK_D: tl.constexpr):
    b = tl.program_id(0)
    n = tl.program_id(1)
    k = tl.program_id(2)
    h = tl.program_id(3)

    acc = 0.0
    # Loop over D in chunks of BLOCK_D
    for d_start in range(0, D, BLOCK_D):
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        mask_d = d_offsets < D
        # Compute input offsets: inp[b, n, k, h, d]
        inp_off = b * inp_stride_b + n * inp_stride_n + k * inp_stride_k + h * inp_stride_h + d_offsets * inp_stride_d
        vals = tl.load(inp_ptr + inp_off, mask=mask_d, other=0.0)
        # Accumulate sum across the chunk (convert boolean mask to float)
        acc += tl.sum(vals, axis=0)

    # Store to output: out[b, n, k, h]
    out_off = b * out_stride_b + n * out_stride_n + k * out_stride_k + h * out_stride_h
    tl.store(out_ptr + out_off, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Triton-only forward: reduce over the last dimension (head_dim) of hidden_states.
        Input:
          hidden_states: [B, N, K, H, D] (float32/float16/bfloat16), device must be CUDA
        Output:
          out: [B, N, K, H] in float32
        """
        # Ensure we are on CUDA and Triton can run
        assert hidden_states.is_cuda, "ModelNew requires CUDA tensors"
        assert hidden_states.dim() == 5, "hidden_states must be 5D [B, N, K, H, D]"
        B_dim, N_dim, K_dim, H_dim, D_dim = hidden_states.shape

        # Prepare input as float32 for computation stability
        inp = hidden_states.to(torch.float32)

        # Allocate output [B, N, K, H] in float32
        out = torch.empty((B_dim, N_dim, K_dim, H_dim), device=hidden_states.device, dtype=torch.float32)

        # Launch Triton kernel: grid over (B, N, K, H)
        grid = (B_dim, N_dim, K_dim, H_dim)
        reduce_last_dim_kernel[grid](
            inp, out,
            B_dim, N_dim, K_dim, H_dim, D_dim,
            inp.stride(0), inp.stride(1), inp.stride(2), inp.stride(3), inp.stride(4),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            BLOCK_D=128  # vectorize over D; supports D up to 128 in chunks
        )

        return out


def run(*args):
    return ModelNew()(*args)
