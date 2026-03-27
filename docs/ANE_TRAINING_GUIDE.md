# ANE Training Implementation Guide

## Architecture Overview

Based on maderix/ANE project analysis, ANE training uses a hybrid approach:

### Compute Distribution
- **ANE**: Forward pass + backward dx (input gradients)
- **CPU**: Backward dW (weight gradients) via cblas + Adam optimizer

### Key Techniques

#### 1. Dynamic Weight Packing
Avoid ANE recompilation (~119 limit) by packing weights into input spatial dimension:

```
Input tensor: [1, C, 1, S + W]
  [0:S]      = activations
  [S:S+W]    = weights (packed spatially)
```

MIL operations:
```mil
tensor<int32, [4]> b_w = const()[val=tensor<int32, [4]>([0,0,0,S])];
tensor<int32, [4]> s_w = const()[val=tensor<int32, [4]>([1,C,1,W])];
tensor<fp16, [1,C,1,W]> W = slice_by_size(x=input,begin=b_w,size=s_w);
```

#### 2. Size Constraints
- **Minimum**: 32x32 = 1,024 elements
- **Maximum**: 16,384 elements total
- **Format**: NCHW [1, C, 1, S] for 1D sequences

#### 3. Tiling Strategy
For operations exceeding 16,384 elements:
- Split into chunks along channel or spatial dimension
- Process each chunk separately
- Concatenate results

#### 4. Multi-Input Constraint
ANE fails with 0x1d error for multiple inputs. Solution:
- Pack all inputs into single spatial tensor
- Use slice_by_size to extract inside kernel

## Implementation Phases

### Phase 1: Basic ANE Operations ✓
- [x] MIL syntax with program(1.3) wrapper
- [x] Empty dictionary for weights (not null)
- [x] Size validation (32x32 min, 16,384 max)
- [x] Basic compile/eval working

### Phase 2: Dynamic Weights
- [ ] Pack weights into spatial dimension
- [ ] slice_by_size extraction
- [ ] Weight update without recompilation

### Phase 3: Training Loop
- [ ] Forward pass on ANE
- [ ] Backward dx on ANE
- [ ] dW gradients on CPU (cblas)
- [ ] Adam optimizer on CPU
- [ ] Parameter golf integration

### Phase 4: Optimizations
- [ ] Tiling for large tensors
- [ ] GQA (Grouped Query Attention)
- [ ] INT8 quantization
- [ ] exec() restart for 119 limit

## Performance Targets

Based on maderix/ANE results:
- Stories110M (109M params): ~91ms/step
- Qwen3-0.6B (596M params): ~412ms/step
- ANE peak: 18.6 TOPS FP16, 35.1 TOPS INT8

## MIL Pattern Examples

### Simple Matmul with Dynamic Weights
```mil
program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}})]
{
  func main<ios18>(tensor<fp16, [1, IC, 1, SEQ+OC]> x) {
    // Slice activations
    tensor<int32, [4]> ba = const()[val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [4]> sa = const()[val=tensor<int32, [4]>([1,IC,1,SEQ])];
    tensor<fp16, [1,IC,1,SEQ]> act = slice_by_size(x=x,begin=ba,size=sa);
    
    // Slice weights
    tensor<int32, [4]> bw = const()[val=tensor<int32, [4]>([0,0,0,SEQ])];
    tensor<int32, [4]> sw = const()[val=tensor<int32, [4]>([1,IC,1,OC])];
    tensor<fp16, [1,IC,1,OC]> W = slice_by_size(x=x,begin=bw,size=sw);
    
    // Matmul
    bool bF = const()[val=bool(false)];
    tensor<int32, [4]> rsh = const()[val=tensor<int32, [4]>([1,1,IC,SEQ])];
    tensor<fp16, [1,1,IC,SEQ]> a2 = reshape(shape=rsh,x=act);
    tensor<fp16, [1,1,SEQ,IC]> a3 = transpose(perm=pm,x=a2);
    tensor<int32, [4]> rsh = const()[val=tensor<int32, [4]>([1,1,IC,OC])];
    tensor<fp16, [1,1,IC,OC]> W2 = reshape(shape=rsh,x=W);
    tensor<fp16, [1,1,SEQ,OC]> yh = matmul(x=a3,y=W2);
    tensor<fp16, [1,1,OC,SEQ]> yt = transpose(perm=pm,x=yh);
    tensor<int32, [4]> ro = const()[val=tensor<int32, [4]>([1,OC,1,SEQ])];
    tensor<fp16, [1,OC,1,SEQ]> out = reshape(shape=ro,x=yt);
  } -> (out);
}
```

## Next Steps

1. Implement dynamic weight packing in `mil_generator.rs`
2. Create `ANETrainer` struct managing ANE+CPU hybrid execution
3. Integrate with parameter-golf training loop
4. Add tiling for 768x256 and larger operations
