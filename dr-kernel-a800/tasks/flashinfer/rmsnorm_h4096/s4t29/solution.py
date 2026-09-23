class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure CUDA tensors and contiguity
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be on CUDA for Triton."
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()
        B, H = hidden.shape
        assert H == 4096, "This kernel expects hidden size 4096."

        # Output tensor same dtype as input hidden
        out = torch.empty_like(hidden)

        # Launch one program per row; parameters that performed best in your environment
        grid = (B,)
        _layernorm_weight_scale_kernel[grid](
            hidden, weight, out,
            B, H, 1e-5,
            hidden.stride(0), hidden.stride(1),
            out.stride(0), out.stride(1),
            num_warps=8, num_stages=2,
        )
        return out


def run(*args):
    return ModelNew()(*args)
