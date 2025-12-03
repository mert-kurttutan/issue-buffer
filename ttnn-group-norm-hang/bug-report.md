# Bug Report: ttnn.group_norm Hangs at Device Synchronization

## Summary
`ttnn.group_norm` operation does not complete execution and hangs indefinitely at the `ttnn.synchronize_device(device)` call.

## Environment

### Software Environment
- **Framework**: TTNN (TensorTorrent Neural Network library)
- **Build Method**: Built from source using `./build_metal.sh`
- **Git Commit**: `8fc61371702e3e653a758e1616a6c3af5637d55b`
- **tt-smi Version**: 3.0.38
- **pyluwen Version**: 0.7.1
- **Python Version**: 3.8.10
- **Data Type**: BFLOAT16
- **Layout**: ROW_MAJOR_LAYOUT

### Hardware Environment
- **Device Type**: Wormhole (n300)
- **Board Configuration**: n300 L + n300 R (dual chip)
- **Board ID**: 100014611916070
- **PCI Device ID**: 0 (device_id=0 used in code)
- **Bus ID**: 0000:02:00.0
- **PCIe**: Gen4 x16

### System Information
- **OS**: Ubuntu 24.04.3 LTS
- **Kernel**: 6.1.62-tenstorrent-gpu
- **Platform**: x86_64
- **Memory**: 31.40 GB
- **Driver**: TT-KMD 1.32

### Firmware Versions
- **TT Flash**: 80.17.0.0
- **CM FW**: 2.32.0.0 (2025-04-11)
- **ETH FW**: 6.14.0
- **BM BL FW**: 129.2.0.0
- **BM APP FW**: 5.12.0.0
- **ARC0/1/3 FW**: 2.32.0.0

### Device Status (at time of report)
- **DRAM Status**: Operational (12G speed)
- **AICLK**: 1000 MHz
- **Voltage**: 0.89V (Left chip), 0.91V (Right chip)
- **Temperature**: 62.2°C (Left chip), 44.2°C (Right chip)
- **Power**: 27W (Left chip), 20W (Right chip)

## Bug Description
The program executes successfully up to and including the `ttnn.group_norm` call, but hangs indefinitely when attempting to synchronize the device. No error messages are produced; the execution simply blocks at synchronization.

## Steps to Reproduce

```python
import ttnn
import torch
import torch.nn.functional as F

# 1. Initialize device
device = ttnn.open_device(device_id=0)

# 2. Define tensor shapes
N, C, H, W = 1, 64, 16, 16
N_G = 8  # Number of groups

# 3. Create PyTorch reference tensors
x_torch = torch.ones((N, C, H, W))
w_torch = torch.ones(C)
b_torch = torch.ones(C)

# 4. Convert input tensor to TTNN format
# Permute from NCHW to NHWC, then reshape to [N, 1, H*W, C]
x_ttnn = ttnn.from_torch(
    x_torch.permute(0, 2, 3, 1).reshape(N, 1, H*W, C),
    dtype=ttnn.DataType.BFLOAT16,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=device,
)

# 5. Create weight tensor with group_norm specific formatting
w_ttnn = ttnn.from_torch(
    ttnn.create_group_norm_weight_bias_rm(
        input_tensor=w_torch,
        num_channels=C,
        num_cores_x=1
    ),
    dtype=ttnn.DataType.BFLOAT16,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=device,
)

# 6. Create bias tensor with group_norm specific formatting
b_ttnn = ttnn.from_torch(
    ttnn.create_group_norm_weight_bias_rm(
        input_tensor=b_torch,
        num_channels=C,
        num_cores_x=1
    ),
    dtype=ttnn.DataType.BFLOAT16,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=device,
)

# 7. Print tensor shapes for verification
print(f"x_ttnn shape: {x_ttnn.shape}")  # Expected: [1, 1, 256, 64]
print(f"w_ttnn shape: {w_ttnn.shape}")
print(f"b_ttnn shape: {b_ttnn.shape}")

# 8. Compute PyTorch reference (completes successfully)
y_torch = F.group_norm(x_torch, N_G, w_torch, b_torch)
print(f"y_torch shape: {y_torch.shape}")

# 9. Create input mask (not passed to group_norm in original code)
input_mask_tensor = ttnn.create_group_norm_input_mask(
    num_channel=C,
    num_groups=N_G,
    num_cores_across_channel=1,
    data_type=ttnn.DataType.BFLOAT16,
)

# 10. Execute TTNN group_norm (this call completes)
y_ttnn = ttnn.group_norm(
    input_tensor=x_ttnn,
    num_groups=N_G,
    weight=w_ttnn,
    bias=b_ttnn,
    core_grid=ttnn.CoreGrid(x=1, y=1),
    inplace=False,
)

# 11. Synchronize device - HANGS HERE INDEFINITELY
ttnn.synchronize_device(device)  # ← Execution blocks at this line

# 12. This line is never reached
print(f"y_ttnn shape: {y_ttnn.shape}")
```

## Expected Behavior
- `ttnn.synchronize_device(device)` should complete and return control
- Final print statement should execute showing the output tensor shape
- Program should terminate normally

## Actual Behavior
- Program hangs indefinitely at `ttnn.synchronize_device(device)`
- No error messages or exceptions are raised
- No timeout occurs
- Process must be manually killed (Ctrl+C or SIGTERM)

## Configuration Details

### Input Parameters
- **Batch size (N)**: 1
- **Channels (C)**: 64
- **Height (H)**: 16
- **Width (W)**: 16
- **Number of groups (N_G)**: 8
- **Input tensor shape (TTNN)**: `[1, 1, 256, 64]` (after reshape from NHWC)
- **Core grid**: `(x=1, y=1)` - minimal configuration
- **Inplace**: `False`

### Tensor Shapes
```
x_ttnn shape: [1, 1, 256, 64]  # [N, 1, H*W, C]
w_ttnn shape: [variable]        # After create_group_norm_weight_bias_rm
b_ttnn shape: [variable]        # After create_group_norm_weight_bias_rm
y_torch shape: [1, 64, 16, 16]  # PyTorch reference output
```

## Additional Notes

1. **Input mask tensor**: The code creates `input_mask_tensor` using `ttnn.create_group_norm_input_mask` but doesn't pass it to `ttnn.group_norm`. The API signature and documentation should clarify if this parameter is required or optional.

2. **Memory configuration**: The sharded memory configuration is commented out in the original code - may need to be specified for proper operation.

3. **Core grid**: Using minimal core grid configuration `(x=1, y=1)`.

4. **PyTorch reference**: The PyTorch `F.group_norm` completes successfully with the same parameters, suggesting the mathematical operation is valid.

5. **Multi-chip configuration**: The system has two Wormhole chips (n300 L and n300 R), but only device_id=0 (left chip with PCIe) is being used. Unclear if this dual-chip configuration affects operation.

6. **No error logging**: No error messages, warnings, or device-level faults reported. Consider adding verbose logging or tracing to identify where the hang occurs in the hardware/firmware stack.

## Questions for Investigation

1. Is the `input_mask_tensor` required to be passed to `ttnn.group_norm`? If so, what parameter name should be used?

2. Should a specific memory configuration (e.g., sharded with `ShardStrategy`, `TensorMemoryLayout.HEIGHT_SHARDED`) be used for group_norm operations?

3. Are there specific constraints on:
   - Tensor shapes (H*W dimensions, channel counts)?
   - Group counts (must divide channels evenly, power of 2)?
   - Core grid configurations (minimum/maximum values)?
   - Device selection in multi-chip configurations?

4. Is there a hardware-level deadlock, resource contention, or command queue issue?

5. Does the operation require explicit memory barriers, fence operations, or completion callbacks before synchronization?

6. Are there any known issues with:
   - Commit hash `8fc61371702e3e653a758e1616a6c3af5637d55b`?
   - Firmware version combinations listed above?
   - Wormhole n300 dual-chip configurations?

7. Should device profiling or tracing be enabled to capture more diagnostic information?

## Workarounds Attempted
None successful - synchronization is mandatory to retrieve results from device.

## Reproduction Rate
- **Frequency**: 100% reproducible with the provided code and parameters on this hardware configuration

## Suggested Debug Steps

1. Enable TTNN verbose logging or trace mode if available
2. Check device command queue status before and after `group_norm` call
3. Verify if other operations successfully synchronize on this device
4. Test with different tensor sizes, group counts, and core grid configurations
5. Test with explicit memory configuration (sharded vs. interleaved)
6. Test passing `input_mask_tensor` to `group_norm` if parameter exists
7. Check for firmware compatibility matrix for this tt-metal version
