class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure inputs are float32 for statistics; no tensor math in host.
        # We convert the view to float32 but do not copy if it already is float32.
        if inputs.dtype != torch.float32:
            inputs_f32 = inputs.to(torch.float32)
        else:
            inputs_f32 = inputs

        B, S, H = inputs_f32.shape

        # Allocate per-row mean and std (fp32) buffers
        mean = torch.empty(B * S, dtype=torch.float32, device=inputs_f32.device)
        std = torch.empty(B * S, dtype=torch.float32, device=inputs_f32.device)

        # Launch compute_mean_std_fp32 kernel
        grid_stats = (B * S,)
        compute_mean_std_fp32[grid_stats](
            inputs_f32, mean, std, B, S, H, BLOCK=2048,
            num_warps=4, num_stages=2
        )

        # Compute icdf for target_sparsity in device, Triton kernel
        z_buf = torch.empty(1, dtype=torch.float32, device=inputs_f32.device)
        compute_icdf_scalar[(1,)](
            z_buf, target_sparsity,
            num_warps=1, num_stages=1
        )
        z = z_buf[0]  # fp32 scalar on device

        # Prepare output buffer in bfloat16
        output = torch.empty_like(inputs_f32, dtype=torch.bfloat16)

        # Launch apply kernel
        apply_threshold_relu_to_bf16[grid_stats](
            inputs_f32, output, mean, std, B, S, H, z, BLOCK=2048,
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
