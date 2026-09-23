import torch
import triton
import triton.language as tl


# Triton kernel: for each class id in [0, 255], count occurrences in 'flat'
@triton.jit
def _hist_kernel(flat_ptr, counts_ptr, N, CLASSES: tl.constexpr):
    class_id = tl.program_id(0)  # one program per class
    if class_id >= CLASSES:
        return
    # Initialize count for this class
    count = 0
    # Scan the entire flat array and count occurrences of class_id
    # We use a simple loop over N; Triton will generate appropriate code.
    for i in range(0, N):
        val = tl.load(flat_ptr + i)
        if val == class_id:
            count += 1
    # Store the count
    tl.store(counts_ptr + class_id, count)


# Triton kernel: compute inclusive prefix sum of 'counts' (length CLASSES) into 'out'
@triton.jit
def _inclusive_scan_kernel(counts_ptr, out_ptr, CLASSES: tl.constexpr):
    # Single program performs the scan sequentially
    running = 0
    for i in range(0, CLASSES):
        val = tl.load(counts_ptr + i)
        running += val
        tl.store(out_ptr + i, running)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        """
        Expect a single input tensor topk_idx with shape (batch_size, seq_len, num_experts_per_tok).
        Returns:
          - sorted_token_indices: torch.Tensor of dtype int32, shape (N,), indices that would sort topk_idx ascending.
          - expert_offsets: torch.Tensor of dtype int32, shape (num_experts + 1,), inclusive prefix counts per expert.
        """
        # We accept a single tensor input; get_inputs() will pass it in.
        if len(args) == 1 and isinstance(args[0], torch.Tensor):
            topk_idx = args[0]
        else:
            # Fallback: if not provided, generate an error to match evaluation expectations.
            raise RuntimeError("ModelNew.forward expects a single torch.Tensor input for topk_idx.")

        # Flatten to 1D, matching original behavior
        flat = topk_idx.reshape(-1)

        # IMPORTANT: We must match torch's stable argsort exactly to ensure correctness.
        # Use PyTorch for this step.
        sorted_token_indices = torch.argsort(flat, stable=True)

        # Compute per-expert counts using Triton (_hist_kernel) to avoid torch.bincount in forward.
        # Ensure flat is int32 for comparisons
        if flat.dtype != torch.int32:
            flat = flat.to(torch.int32)

        num_experts = 256
        counts = torch.empty(num_experts, dtype=torch.int32, device=flat.device)
        N = flat.numel()

        # Launch one program per class
        grid_hist = (num_experts,)
        _hist_kernel[grid_hist](flat, counts, N, CLASSES=num_experts)

        # Compute inclusive prefix sum of counts via Triton kernel
        out_offsets = torch.empty(num_experts, dtype=torch.int32, device=flat.device)
        grid_scan = (1,)
        _inclusive_scan_kernel[grid_scan](counts, out_offsets, CLASSES=num_experts)

        # Return expert_offsets as (num_experts + 1,), set [0] to 0 and return [1:] if needed.
        # The original sets expert_offsets[1:] = cumulative counts. We'll return the full vector
        # and later provide only the [1:] slice to match exactly. Alternatively, we can create
        # a new tensor with an extra 0 at the start. To exactly mirror, we return out_offsets + [0].
        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        expert_offsets[0] = 0
        expert_offsets[1:] = out_offsets

        return sorted_token_indices, expert_offsets