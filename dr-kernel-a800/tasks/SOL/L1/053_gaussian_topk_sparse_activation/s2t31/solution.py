class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity requested, return input unchanged
        if target_sparsity == 0.0:
            return x

        # Ensure contiguous and compute in float32 for stability
        x_f32 = x.contiguous().to(torch.float32)
        B, S, N = x_f32.shape

        # Output as float32; we'll cast to bfloat16 at the end
        out_f32 = torch.empty((B, S, N), dtype=torch.float32, device=x_f32.device)

        # 1-element buffer for inv-Phi scalar (float32)
        std_multiplier = torch.empty((1,), dtype=torch.float32, device=x_f32.device)

        # Launch Triton kernel: one program per row
        grid = (B * S,)
        sparsity_row_kernel[grid](
            x_f32, out_f32, std_multiplier, N, float(target_sparsity),
            BLOCK_SIZE=1024, num_warps=8
        )

        # Return in bfloat16 to match typical behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
