import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_rows_kernel_fp32(
    out_ptr,          # *fp32, shape (N, H)
    src_ptr,          # *fp32, shape (M, H)
    indices_ptr,      # *int32, shape (M,)
    N: tl.constexpr,  # number of rows (batch_seq_len)
    H: tl.constexpr,  # hidden_size
):
    # 1D grid: each program handles one row
    pid = tl.program_id(0)
    if pid >= N:
        return

    # Load destination row index for this token
    row_idx = tl.load(indices_ptr + pid)  # int32

    # Loop over hidden dimension; perform atomic add per element
    for j in range(0, H):
        # Compute source and destination offsets in row-major layout
        src_off = pid * H + j
        dst_off = row_idx * H + j

        # Load source element (fp32) and atomic add to destination
        val = tl.load(src_ptr + src_off)  # fp32
        tl.atomic_add(out_ptr + dst_off, val)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized scatter-add:
        out[token_indices[i]] += expert_outputs[i] for all i.
        We clone final_hidden_states to match the reference behavior, perform accumulation in fp32,
        and cast back to bfloat16 for the return.
        """
        # Ensure tensors are on CUDA
        assert final_hidden_states.is_cuda, "final_hidden_states must be on CUDA"
        assert expert_outputs.is_cuda, "expert_outputs must be on CUDA"
        assert token_indices.is_cuda, "token_indices must be on CUDA"

        # Clone to match reference (clone before index_add_)
        out = final_hidden_states.clone()
        out = out.contiguous()

        # Cast to fp32 for robust atomic_add in Triton
        out_fp32 = out.to(torch.float32)
        expert_outputs_fp32 = expert_outputs.to(torch.float32).contiguous()

        # Shapes
        N = out_fp32.shape[0]   # batch_seq_len
        H = out_fp32.shape[1]   # hidden_size
        M = expert_outputs_fp32.shape[0]  # number of selected tokens

        # Triton prefers int32 for indices
        indices_i32 = token_indices.to(torch.int32)

        # Launch Triton kernel: 1D grid over rows
        grid = (N,)
        scatter_add_rows_kernel_fp32[grid](
            out_fp32, expert_outputs_fp32, indices_i32,
            N=N, H=H,
            num_warps=4, num_stages=2
        )

        # Cast back to bfloat16 to match original dtype
        out_bf16 = out_fp32.to(torch.bfloat16)
        return out_bf16


def run(*args):
    return ModelNew()(*args)
