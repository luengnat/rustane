# ANE MIL Working Format (from ane-lora-training)

## Key Requirements

### 1. **Use fp32 I/O with cast to fp16**
```mil
func main<ios18>(tensor<fp32, [1, 512, 1, 1024]> x) {
    // Cast input to fp16
    string to_fp16 = const()[name = string("to_fp16"), val = string("fp16")];
    tensor<fp16, [1, 512, 1, 1024]> x16 = cast(dtype = to_fp16, x = x);
    
    // ... operations on fp16 ...
    
    // Cast output back to fp32
    string to_fp32 = const()[name = string("to_fp32"), val = string("fp32")];
    tensor<fp32, [1, 512, 1, 1024]> y = cast(dtype = to_fp32, x = y16);
} -> (y);
```

### 2. **Weight Blob Format (128 bytes)**
```
[0:4]   version = 1
[4:8]   type = 2 (fp16)
[8:64]  reserved zeros

[64:68] magic = 0xDEADBEEF (LE)
[68:72] chunk_count = 1
[72:76] data_size (fp16 byte count)
[76:80] reserved
[80:84] data_offset = 128
[84:128] reserved zeros

[128:] fp16 weight data
```

### 3. **Spatial Dimension Must Be >= 16 AND Multiple of 16**
- 8, 12, 24 ❌ (fail at eval)
- 16, 32, 48 ✅ (work)

### 4. **Use ONLY conv (NO matmul)**
```mil
# ❌ BAD - matmul fails at eval
tensor<fp16, [...]> y = matmul(x = a, y = b);

# ✅ GOOD - conv works
tensor<fp16, [...]> y = conv(dilations = dl, groups = gr, pad = pd, 
                              pad_type = pt, strides = st, weight = W, x = x);
```

### 5. **Build Info Required**
```
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, 
    {"coremlc-version", "3505.4.1"}, 
    {"coremltools-component-milinternal", ""}, 
    {"coremltools-version", "9.0"}})]
```

## Working Example

See: `ane-lora-training/ane_lora_kernels.py` lines 149-166

## Key Differences from Our MIL

| Issue | Our MIL | Working MIL |
|-------|---------|-------------|
| I/O dtype | fp16 | fp32 (cast internally) |
| Weight header | 64 bytes | 128 bytes |
| Spatial dim | 1024 ✅ | Must pad if <16 or not ×16 |
| Operations | matmul, reduce_mean | conv only |
| Cast ops | Missing | Required fp32↔fp16 |

## Fix Strategy

1. Change all I/O to fp32
2. Add cast operations
3. Replace matmul with conv
4. Fix weight blob headers to 128 bytes
5. Remove reduce_mean (use conv-based normalization or skip)
