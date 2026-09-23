import torch
import triton

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Expect inputs: hidden_states [batch, 4096], weight [4096], possibly in bfloat16
        assert hidden_states.dim() == 2 and hidden_states.shape[1] == 4096, "hidden_states must be [batch, 4096]"
        assert weight.dim() == 1 and weight.shape[0] == 4096, "weight must be [4096]"

        # Ensure contiguous and CUDA tensors
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()
        device = hidden.device
        assert hidden.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"

        batch_size = hidden.shape[0]
        hidden_size = hidden.shape[1]

        # Output tensor (float32 inside kernel, cast back to original dtype)
        out = torch.empty_like(hidden, dtype=torch.float32, device=device)
        inv_rms = torch.empty(batch_size, dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per row, vectorize across 4096 columns
        grid = (batch_size,)
        _row_fused_scale_kernel[grid](
            hidden, weight, inv_rms, out,
            hidden_size=hidden_size,
            EPS=1e-5,
            BLOCK_SIZE=hidden_size,
            num_warps=16,  # robust configuration for 4096-wide tiles
            num_stages=2
        )

        # Cast back to original dtype to match the original model's behavior
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
