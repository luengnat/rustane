# ANE IOSurface Best Practices

## Overview

Based on analysis of ANEMLL and Espresso projects, this document describes the correct way to use IOSurface for ANE (Apple Neural Engine) operations.

## Why IOSurface?

**Critical for ANE Synchronization:**
- ANE hardware writes to memory asynchronously
- Without IOSurface, CPU reads can see stale/corrupted data
- IOSurface provides proper memory coherency between ANE and CPU

From ANEMLL (v0.3.5):
> "All pixel buffer outputs now use IOSurface backing for proper ANE synchronization. This eliminates coherency hazards between the ANE hardware and CPU reads without large sleep delays."

## IOSurface Creation

### Basic Creation (Espresso)

```objc
IOSurfaceRef ane_interop_create_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0
    });
}
```

### Key Properties:
- **Width**: Set to total bytes (treat as 1D array)
- **Height**: Always 1 (we're using 1D layout)
- **BytesPerElement**: 1 (byte-level access)
- **BytesPerRow**: Same as width (no padding)
- **AllocSize**: Total allocation size
- **PixelFormat**: 0 (raw data)

## Lock/Unlock Pattern

### Critical for Correctness

**ALWAYS lock before CPU access, unlock after:**

```objc
// Lock for reading
IOSurfaceLock(surface, .readOnly, nil);
defer { IOSurfaceUnlock(surface, .readOnly, nil); }

// Get base address
void *baseAddress = IOSurfaceGetBaseAddress(surface);

// Read/write data...
```

### Lock Options:

| Option | Use Case |
|--------|----------|
| `.readOnly` | CPU only reading data |
| `[]` (empty) | CPU writing data |
| `.readOnly` + `kIOSurfaceLockAvoidSync` | Avoid expensive cache flush (advanced) |

## Buffer Management Strategies

### 1. Simple Pool (ANEMLL-style)

```swift
// Pre-allocate multiple buffers
let bufferPool: [IOSurfaceRef] = (0..<4).map { _ in
    createIOSurface(size: bufferSize)
}

// Use round-robin to avoid reuse
let buffer = bufferPool[tokenIndex % bufferPool.count]
```

### 2. Ping-Pong Buffers (Espresso-style)

```swift
// Two buffer sets for streaming inference
var pingBuffers: [IOSurfaceRef] = []
var pongBuffers: [IOSurfaceRef] = []
var usePing = true

func getBuffers() -> (input: IOSurfaceRef, output: IOSurfaceRef) {
    let buffers = usePing ? pingBuffers : pongBuffers
    usePing.toggle()
    return (buffers[0], buffers[1])
}
```

**Why ping-pong?** Prevents ANE from reading a buffer that's being written to by the CPU.

### 3. Ring Buffer (ANEMLL-style)

```swift
// N-deep ring for monolithic models
let ringDepth = 16
var ringBuffers: [IOSurfaceRef] = []
var tokenCounter = 0

func getBuffer() -> IOSurfaceRef {
    let index = tokenCounter % ringDepth
    tokenCounter += 1
    return ringBuffers[index]
}
```

**Why ring?** Ensures buffer isn't reused while still being read (16 tokens of safety margin).

## Serial Execution Queue

### Critical for Thread Safety

From Espresso:
```swift
// Serial queue for ANE predictions
private let predictionQueue = DispatchQueue(
    label: "com.anemll.prediction",
    qos: .userInitiated
)

// All ANE operations run on this queue
predictionQueue.async {
    // Compile, eval, state reads
}
```

**Why serial?** ANE + MLState may not be thread-safe when accessed from different threads.

## Complete Workflow Example

```swift
import IOSurface

class ANEExecutor {
    // Pre-allocated surfaces
    var inputSurfaces: [IOSurfaceRef] = []
    var outputSurfaces: [IOSurfaceRef] = []
    
    // Serial queue
    let queue = DispatchQueue(label: "ane.exec")
    
    func execute(inputData: [Float]) -> [Float] {
        return queue.sync {
            // 1. Get surfaces
            let inputSurf = inputSurfaces[currentIndex]
            let outputSurf = outputSurfaces[currentIndex]
            
            // 2. Write input (lock → write → unlock)
            IOSurfaceLock(inputSurf, [], nil)
            let inputPtr = IOSurfaceGetBaseAddress(inputSurf)
            memcpy(inputPtr, inputData, inputData.count * 4)
            IOSurfaceUnlock(inputSurf, [], nil)
            
            // 3. Execute on ANE
            aneKernel.eval(
                inputSurface: inputSurf,
                outputSurface: outputSurf
            )
            
            // 4. Read output (lock → read → unlock)
            IOSurfaceLock(outputSurf, .readOnly, nil)
            let outputPtr = IOSurfaceGetBaseAddress(outputSurf)
            var output = Array(repeating: Float(0), count: outputSize)
            memcpy(&output, outputPtr, outputSize * 4)
            IOSurfaceUnlock(outputSurf, .readOnly, nil)
            
            return output
        }
    }
}
```

## Common Pitfalls

### 1. **Missing Lock/Unlock**
```swift
// ❌ WRONG
let ptr = IOSurfaceGetBaseAddress(surface)
readData(ptr)  // May read stale data!

// ✅ CORRECT
IOSurfaceLock(surface, .readOnly, nil)
let ptr = IOSurfaceGetBaseAddress(surface)
readData(ptr)
IOSurfaceUnlock(surface, .readOnly, nil)
```

### 2. **Buffer Reuse Race**
```swift
// ❌ WRONG - May overwrite buffer still being read
let output = surfaces[0]
kernel.eval(output)
// ... immediately use output for next token without checking completion

// ✅ CORRECT - Use ring buffer or ping-pong
let output = ringBuffers[tokenCounter % 16]
kernel.eval(output)
tokenCounter += 1  // Won't reuse for 16 tokens
```

### 3. **Multi-threaded Access**
```swift
// ❌ WRONG - Race condition
DispatchQueue.global().async { kernel.eval() }
DispatchQueue.global().async { kernel.eval() }

// ✅ CORRECT - Serial queue
predictionQueue.async { kernel.eval() }
```

## Performance Considerations

### Lock Overhead
- Lock/Unlock has small overhead (~microseconds)
- Batch operations when possible
- Use `kIOSurfaceLockAvoidSync` for read-only if you know ANE is done

### Buffer Pool Sizing
- **Minimum**: 2 buffers (ping-pong)
- **Recommended**: 4-8 buffers
- **Monolithic models**: 16 buffers (ring buffer)

### Memory Alignment
- IOSurface is automatically aligned for ANE
- Manual buffers should be 16-byte aligned
- Use `posix_memalign` or Swift's `UnsafeMutableRawPointer.allocate`

## Comparison: With vs Without IOSurface

| Aspect | Without IOSurface | With IOSurface |
|--------|------------------|----------------|
| **Correctness** | Race conditions, stale data | Reliable synchronization |
| **Latency** | Fast (no lock) | Slightly slower (lock/unlock) |
| **Complexity** | Simple | Requires lock discipline |
| **ANE Support** | May work for simple cases | Required for reliable operation |

## References

- ANEMLL: https://github.com/Anemll/Anemll
- Espresso: https://github.com/christopherkarani/Espresso
- Apple IOSurface docs: https://developer.apple.com/documentation/iosurface

## Key Takeaways

1. **Always use IOSurface** for ANE I/O buffers
2. **Always lock/unlock** before CPU access
3. **Use serial queue** for all ANE operations
4. **Use ping-pong or ring buffers** to prevent reuse races
5. **Pre-allocate** buffers to avoid allocation overhead during inference
