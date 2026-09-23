import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: per-source-row accumulation along hidden dimension, explicit bf16 handling.
# Each program handles one row i and loops over H, updating output[token_indices[i], h] += expert_outputs[i, h].
if TRITON_AVAILABLE:
    @triton.jit
    def scatter_add_rows_bf16_kernel(
        output_ptr,          # *const bfloat16
        expert_ptr,          # *const bfloat16
        indices_ptr,         # *const int64
        B: tl.constexpr,     # batch_seq_len (rows in output)
        H: tl.constexpr,     # hidden_size (columns)
        T: tl.constexpr,     # number of expert outputs
    ):
        # One program per source row i
        i = tl.program_id(0)
        if i >= T:
            return

        # Load the destination row index (int64) for this source row
        idx64 = tl.load(indices_ptr + i)

        # Loop over hidden dimension
        for h in range(0, H):
            # Compute linear offsets
            out_off = idx64.to(tl.int32) * H + h
            exp_off = i * H + h

            # Load v from expert_outputs as bfloat16
            v = tl.load(expert_ptr + exp_off, eviction_policy='evict_last')

            # Load current output value as bfloat16
            out_val = tl.load(output_ptr + out_off, eviction_policy='evict_last')

            # Add in bfloat16
            out_val = out_val + v

            # Store back (bfloat16)
            tl.store(output_ptr + out_off, out_val, eviction_policy='evict_last')


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized version of:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        We implement the index_add via a Triton kernel using a deterministic per-element update in bfloat16.
        """
        # If Triton not available or tensors not on CUDA, fall back to PyTorch
        if (not TRITON_AVAILABLE) or (not final_hidden_states.is_cuda) or (not expert_outputs.is_cuda) or (not token_indices.is_cuda):
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
            return output

        # Ensure contiguity
        output = final_hidden_states.clone()
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Shapes
        B = output.shape[0]  # batch_seq_len
        H = output.shape[1]  # hidden_size
        T = token_indices.shape[0]  # number of expert outputs

        # Launch Triton kernel: one program per source row
        scatter_add_rows_bf16_kernel[(T,)](
            output, expert_outputs, token_indices,
            B=B, H=H, T=T,
            num_warps=1,  # keep simple and deterministic
            num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
