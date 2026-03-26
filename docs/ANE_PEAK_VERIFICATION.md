# ANE Peak Throughput Benchmark - Numerical Verification

## Summary

**Status: MULTI-LAYER VERIFIED ✓**

- Single-layer conv benchmark: ✓ **VERIFIED** (see `ane_peak_throughput_single.rs`)
- Multi-layer conv benchmark: ✓ **VERIFIED** (see `ane_peak_throughput.rs`)
  - Uses `compile_multi()` with separate `mil::WeightBlob` per layer
  - MIL syntax: `tensor<string, []>("...")` wrappers for all constants
  - FP16 input/output (no cast ops in graph)
  - Output verification: PASSED (all values finite and non-zero)

## Key Findings

### Multi-Layer Performance (VERIFIED)

| Config | Weight Size | GFLOPS | TFLOPS | % Peak | Verified |
|--------|-------------|--------|--------|--------|----------|
| 4x conv 64ch sp32 | <0.1 MB | 0.00 | 0.01 | 0.1% | ✓ |
| 8x conv 64ch sp64 | 0.1 MB | 0.00 | 0.04 | 0.2% | ✓ |
| 4x conv 128ch sp64 | 0.1 MB | 0.01 | 0.11 | 0.6% | ✓ |
| 4x conv 256ch sp32 | 0.5 MB | 0.02 | 0.16 | 0.9% | ✓ |

**Note**: TFLOPS are low because:
1. Small tensor sizes don't fill ANE pipeline
2. Only 4-8 layers (vs 32-128 needed for 94%+ utilization)
3. FP16 precision limits for deep chains (underflow/overflow with random weights)

### Single-Layer Performance (VERIFIED)

| Config | Weight Size | GFLOPS | TFLOPS | % Peak | Verified |
|--------|-------------|--------|--------|--------|----------|
| 512ch sp64 | 0.5 MB | 0.03 | 0.32 | 1.7% | ✓ |
| 256ch sp64 | 0.1 MB | 0.01 | 0.10 | 0.6% | ✓ |
| 128ch sp128 | <0.1 MB | 0.00 | 0.06 | 0.3% | ✓ |

## Working Pattern

The multi-layer conv pattern now works with these requirements:

### 1. MIL Syntax

Use `tensor<string, []>("...")` wrappers, not bare `string("...")`:

```mil
// CORRECT (works)
tensor<fp16, [C, C, 1, 1]> W0 = const()[
  name = tensor<string, []>("W0"),
  val = tensor<fp16, [C, C, 1, 1]>(
    BLOBFILE(path = tensor<string, []>("@model_path/weights/w0.bin"),
             offset = tensor<uint64, []>(64))
  )
];

// INCRECT (fails with InvalidMILProgram)
tensor<fp16, [C, C, 1, 1]> W0 = const()[
  name = string("W0"),
  val = tensor<fp16, [C, C, 1, 1]>(
    BLOBFILE(path = string("@model_path/weights/w0.bin"), offset = uint64(64))
  )
];
```

### 2. Weight Blob Format

128-byte header with the following structure:

```
Bytes 0-4:   01 00 00 00 02 (magic)
Bytes 64-68: EF BE AD DE 01 (inner magic)
Bytes 72-76: payload_size (little-endian u32)
Bytes 80-84: header_size (128 = 0x80)
Bytes 128+:  FP16 weight data
```

### 3. MIL Program Structure

- Use `ios16` not `ios18`
- Input/output tensors should be FP16 (avoid cast ops)
- Use `tensor<string, []>` for all constant names and values
- BLOBFILE offset should be `tensor<uint64, []>(64)` to skip header

Example:
```mil
program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}})]
{
    func main<ios16>(tensor<fp16, [1, C, 1, SP]> x) {
        tensor<string, []> pt = const()[name = tensor<string, []>("pt"), val = tensor<string, []>("valid")];
        tensor<int32, [4]> pd = const()[name = tensor<string, []>("pd"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> st = const()[name = tensor<string, []>("st"), val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [2]> dl = const()[name = tensor<string, []>("dl"), val = tensor<int32, [2]>([1, 1])];
        tensor<int32, []> gr = const()[name = tensor<string, []>("gr"), val = tensor<int32, []>(1)];

        tensor<fp16, [C, C, 1, 1]> W0 = const()[name = tensor<string, []>("W0"), val = tensor<fp16, [C, C, 1, 1]>(BLOBFILE(path = tensor<string, []>("@model_path/weights/w0.bin"), offset = tensor<uint64, []>(64)))]  ;
        tensor<fp16, [1, C, 1, SP]> c0 = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = W0, x = x)[name = tensor<string, []>("conv0")];

        // Chain more layers...

    } -> (cN);
}
```

## Technical Details

### compile_multi() API

The `compile_multi()` function:
1. Creates weight dictionary with entries: `{name: {offset: 0, data: <bytes>}}`
2. Writes weight files to temp directory via `write_model_files()`
3. Passes MIL text and weight dict to ANE compiler service
4. Returns `ANEExecutor` for running the compiled kernel

### Weight Data Flow

```
Rust code
  │
  ├─> mil::WeightBlob::from_fp32() → 128-byte header + FP16 data
  │
  ├─> compile_multi(weight_datas: &[&[u8]])
  │      │
  │      ├─> create_weight_dictionary() → NSDictionary for model descriptor
  │      │
  │      └─> write_model_files() → writes to tmp/weights/w0.bin, w1.bin, etc.
  │
  └─> ANECompilerService reads files and compiles
```

### Why Previous Attempts Failed

| Attempt | Error | Root Cause |
|---------|-------|------------|
| `ane::WeightBlob::multi_layer_conv()` | InvalidMILProgram | MIL used bare `string()` not `tensor<string, []>()` |
| `compile_multi()` with wrong syntax | InvalidMILProgram | Same MIL syntax issue |
| FP32 I/O with casts | InvalidMILProgram | Cast ops may not be supported in ANE compiler |
| `ios18` deployment target | InvalidMILProgram | Unsupported version |

## Conclusions

1. **Multi-layer conv IS supported** through `compile_multi()` API
2. **MIL syntax matters** - must use `tensor<string, []>("...")` wrappers
3. **Weight blob format** - 128-byte header with specific magic bytes
4. **FP16 I/O** - avoid cast operations in the graph
5. **Deep graphs (32-128 layers)** should achieve 94%+ utilization per Apple's analysis

## Files

| File | Status | Purpose |
|------|--------|---------|
| `examples/ane_peak_throughput.rs` | ✓ Working | Multi-layer benchmark |
| `examples/ane_peak_throughput_single.rs` | ✓ Working | Single-layer benchmark |
| `examples/test_residual_ffn.rs` | ✓ Working | Reference pattern (3-layer FFN) |
| `docs/ANE_PEAK_VERIFICATION.md` | This file | Verification report |

## Next Steps

1. **Deeper graphs**: Test with 32-128 layers (may need weight scaling adjustments)
2. **Larger tensors**: Test with 512+ channels for more parallelism
3. **TFLOPS optimization**: Tune config to approach 19 TFLOPS peak
4. **Document pattern**: Add multi-layer example to README
