# ANE MIL Syntax - Working Format

Based on analysis of the working ANE project code.

## Key Differences from rustane

### ❌ rustane (FAILING)
```mil
main f(x: tensor<4xf32>) -> (y: tensor<4xf32>) {
    let y = x + x;
    return (y);
}
```

### ✅ ANE Project (WORKING)
```mil
func main<ios18>(tensor<fp16, [1, 512, 1, 512]> x) {
    tensor<fp16, [1,512,1,512]> out = mul(x=x,y=x)[name=string("out")];
} -> (out);
```

## ANE MIL Format Rules

### 1. Function Declaration
```mil
func main<ios18>(tensor<TYPE, [SHAPE]> x) {
    ...
} -> (output_name);
```

**Required:**
- `func` keyword (not just `main`)
- `<ios18>` tag (ANE iOS version)
- No return type in signature
- `-> (output)` at end

### 2. Tensor Types
```mil
tensor<fp16, [1, 512, 1, 512]>   // 4D ANE layout [N, C, H, W]
tensor<fp32, [1, 768, 1, 256]>   // fp32 variant
tensor<int32, [3]>                // Integer tensor
```

**ANE Layout:** `[batch, channels, height, width]`
- Use `[1, C, 1, S]` for 1D sequences
- Use `[1, C, H, W]` for 2D data

### 3. Operations
```mil
// Binary operations
tensor<fp16, [1,C,1,S]> out = mul(x=a,y=b)[name=string("out")];
tensor<fp16, [1,C,1,S]> out = add(x=a,y=b)[name=string("out")];

// Named operations (required!)
[name=string("unique_name")]

// Constants
fp16 val = const()[name=string("c"), val=fp16(0.5)];
int32 axis = const()[name=string("ax"), val=int32(1)];
bool flag = const()[name=string("f"), val=bool(false)];
```

### 4. Weight Loading
```mil
tensor<fp16, [VOCAB,DIM,1,1]> We = const()[
    name=string("We"),
    val=tensor<fp16, [VOCAB,DIM,1,1]>(
        BLOBFILE(
            path=string("@model_path/weights/embed.bin"),
            offset=uint64(64)
        )
    )
];
```

**Note:** 64-byte offset for blob header

### 5. Complete Example - RMSNorm
```mil
func main<ios18>(tensor<fp16, [1, 768, 1, 256]> x) {
    // Square
    tensor<fp16, [1,768,1,256]> sq = mul(x=x,y=x)[name=string("sq")];
    
    // Sum over channels (axis=1)
    tensor<int32, [1]> rax = const()[name=string("rax"), val=tensor<int32, [1]>([1])];
    bool kd = const()[name=string("kd"), val=bool(true)];
    tensor<fp16, [1,1,1,256]> ss = reduce_sum(x=sq,axes=rax,keep_dims=kd)[name=string("ss")];
    
    // Multiply by 1/DIM
    fp16 invd = const()[name=string("invd"), val=fp16(0.001302)];
    tensor<fp16, [1,1,1,256]> ss2 = mul(x=ss,y=invd)[name=string("ss2")];
    
    // Add epsilon
    fp16 eps = const()[name=string("eps"), val=fp16(0.00001)];
    tensor<fp16, [1,1,1,256]> ss3 = add(x=ss2,y=eps)[name=string("ss3")];
    
    // pow(x, -0.5) = 1/sqrt(x)
    fp16 nhalf = const()[name=string("nhalf"), val=fp16(-0.5)];
    tensor<fp16, [1,1,1,256]> rrms = pow(x=ss3,y=nhalf)[name=string("rrms")];
    
    // Multiply input by reciprocal RMS
    tensor<fp16, [1,768,1,256]> xr = mul(x=x,y=rrms)[name=string("xr")];
    
    // Load weights
    tensor<fp16, [1,768,1,1]> rw = const()[
        name=string("rw"),
        val=tensor<fp16, [1,768,1,1]>(
            BLOBFILE(path=string("@model_path/weights/rms_w.bin"), offset=uint64(64))
        )
    ];
    
    // Final multiply
    tensor<fp16, [1,768,1,256]> out = mul(x=xr,y=rw)[name=string("out")];
} -> (out);
```

## Supported Operations

From working code:
- `mul(x=,y=)` - Element-wise multiply
- `add(x=,y=)` - Element-wise add
- `conv(dilations=,groups=,pad=,strides=,weight=,x=)` - Convolution
- `matmul(transpose_x=,transpose_y=,x=,y=)` - Matrix multiply
- `reduce_sum(x=,axes=,keep_dims=)` - Sum reduction
- `softmax(axis=,x=)` - Softmax
- `pow(x=,y=)` - Power
- `reshape(shape=,x=)` - Reshape

## Weight Blob Format

ANE expects specific blob format with 64-byte header:
```c
struct ANEBlobHeader {
    uint32_t magic;      // 'ANEB'
    uint32_t version;    // 1
    uint32_t data_type;  // 16=fp16, 32=fp32
    uint32_t dims;       // Number of dimensions
    uint32_t shape[8];   // Shape array
    uint32_t reserved[4];
};
// Followed by raw weight data
```

## Why rustane Fails

rustane generates:
```mil
main f(x: tensor<4xf32>) -> (y: tensor<4xf32>) {
    let y = x + x;
    return (y);
}
```

But ANE needs:
```mil
func main<ios18>(tensor<fp16, [1, 4, 1, 1]> x) {
    tensor<fp16, [1,4,1,1]> out = add(x=x,y=x)[name=string("out")];
} -> (out);
```

**Completely different syntax!**

## Recommendation

ANE MIL syntax is proprietary and undocumented. The only reliable way to use ANE is through:
1. **CoreML** (official Apple framework)
2. **Copy ANE project approach exactly** (private APIs, specific MIL dialect)

Direct ANE access requires exact replication of the ANE project's MIL generation approach.
