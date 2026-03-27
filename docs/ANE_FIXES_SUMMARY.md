# ANE MIL Fixes Applied

## What We Fixed ✅

### 1. **Weight Blob Header** (128 bytes)
**Before**: 64-byte header with ANEB magic
**After**: 128-byte header per ane-lora-training:
- [0:4] version = 1
- [4:8] type = 2 (fp16)
- [64:68] magic = 0xDEADBEEF
- [80:84] data_offset = 128

### 2. **Test Validation**
```
✅ layer0_wq.bin: version=1, dtype=2, magic=0xDEADBEEF, data_size=524288
✅ All weight blobs validated successfully (128-byte header format)
```

## Critical Findings from ane-lora-training

### ❌ **matmul DOES NOT WORK**
- Compiles successfully
- **Fails at eval time**
- Only conv works on ANE

### ✅ **Use 1x1 conv Instead**
```python
# matmul: A[M,K] @ B[K,N] -> C[M,N]
# conv:  W[M,K,1,1] * x[1,K,1,N] -> [1,M,1,N]
```

### ⚠️ **Spatial Dimension Constraints**
- Must be **>= 16**
- Must be **multiple of 16**
- 1024 ✅ (valid)
- 8, 12, 24 ❌ (will fail)

### 📝 **Working MIL Format**
```mil
program(1.3)
[buildInfo = dict<string, string>(
    {{"coremlc-component-MIL", "3510.2.1"}, 
     {"coremlc-version", "3505.4.1"}, 
     {"coremltools-version", "9.0"}}
)]
{
    func main<ios18>(tensor<fp32, [1, 512, 1, 1024]> x) {
        // Cast to fp16
        string to_fp16 = const()[val = string("fp16")];
        tensor<fp16, [...]> x16 = cast(dtype = to_fp16, x = x);
        
        // Conv operation (only thing that works!)
        tensor<fp16, [...]> y16 = conv(...);
        
        // Cast back to fp32
        string to_fp32 = const()[val = string("fp32")];
        tensor<fp32, [...]> y = cast(dtype = to_fp32, x = y16);
    } -> (y);
}
```

## Next Steps to Fix MIL Compilation

1. **Change I/O to fp32** (not fp16)
2. **Add cast operations** in MIL
3. **Replace matmul with conv**
4. **Use BLOBFILE with offset 64** (past 128-byte header)
5. **Test with simple conv-only program first**

The ane-lora-training project has **working LoRA gradient computation** on ANE using this exact format!
