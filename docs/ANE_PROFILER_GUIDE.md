# ANE Profiler Guide

## Overview

The ANE Profiler provides comprehensive timing and performance analysis for ANE kernel execution. It helps identify bottlenecks, measure utilization, and optimize training performance.

## Features

1. **Per-kernel timing** - Capture compile, execute, and data transfer times
2. **Performance analytics** - Throughput, utilization, bottleneck analysis
3. **Automatic report generation** - Human-readable performance reports
4. **RAII timing guards** - Automatic timing via `KernelTimer`

## Quick Start

```rust
use rustane::ane::ANEProfiler;

// Create a profiler
let mut profiler = ANEProfiler::new();

// Profile training steps
for step in 0..100 {
    profiler.start_step();

    // Profile individual kernels
    profiler.start_kernel("rmsnorm_layer");
    profiler.start_h2d();
    // ... transfer data to ANE
    profiler.end_h2d();

    // ... execute kernel
    let timing = profiler.end_kernel("rmsnorm_layer", input_bytes, output_bytes);

    profiler.end_step();
}

// Generate performance report
println!("{}", profiler.generate_report());
```

## Using KernelTimer (RAII)

For automatic timing, use the `KernelTimer` guard:

```rust
use rustane::ane::{ANEProfiler, KernelTimer};

let mut profiler = ANEProfiler::new();

profiler.start_step();

{
    let mut timer = KernelTimer::new(&mut profiler, "matmul_kernel");
    timer.with_data(input_bytes, output_bytes);

    // ... execute kernel
    // Timing automatically recorded when timer drops
}

profiler.end_step();
```

## API Reference

### ANEProfiler

#### Creation
- `new()` - Create enabled profiler
- `disabled()` - Create disabled profiler (zero overhead)
- `set_enabled(bool)` - Enable/disable profiling

#### Step Timing
- `start_step()` - Begin profiling a training step
- `end_step()` - End step, returns `StepProfile`

#### Kernel Timing
- `start_kernel(&str)` - Start timing a kernel
- `end_kernel(&str, input_bytes, output_bytes)` - End timing, returns `KernelTiming`
- `start_h2d()` / `end_h2d()` - Time host-to-device transfers
- `start_d2h()` / `end_d2h()` - Time device-to-host transfers
- `record_compile_time(Duration)` - Record compilation time

#### Analytics
- `get_stats()` - Get per-kernel statistics (`Vec<KernelStats>`)
- `get_metrics()` - Get overall metrics (`ProfilerMetrics`)
- `generate_report()` - Generate human-readable report
- `clear()` - Clear all profiling data

### Data Structures

#### KernelTiming
```rust
pub struct KernelTiming {
    pub kernel_name: String,
    pub compile_time: Option<Duration>,
    pub h2d_time: Duration,      // Host to device transfer
    pub exec_time: Duration,     // Kernel execution
    pub d2h_time: Duration,      // Device to host transfer
    pub total_time: Duration,
    pub input_bytes: usize,
    pub output_bytes: usize,
}
```

#### KernelStats
```rust
pub struct KernelStats {
    pub kernel_name: String,
    pub call_count: u64,
    pub total_exec_time: Duration,
    pub total_transfer_time: Duration,
    pub avg_exec_time: Duration,
    pub min_exec_time: Duration,
    pub max_exec_time: Duration,
    pub total_bytes: usize,
    pub throughput_gbps: f64,      // Data processed per second
    pub pct_of_total: f64,         // % of total compute time
}
```

#### ProfilerMetrics
```rust
pub struct ProfilerMetrics {
    pub total_steps: u64,
    pub avg_step_time: Duration,
    pub total_kernels: u64,
    pub total_compute_time: Duration,
    pub total_transfer_time: Duration,
    pub total_compile_time: Duration,
    pub avg_ane_utilization: f64,  // exec_time / step_time
    pub overall_throughput_gbps: f64,
}
```

## Example Report Output

```
╔══════════════════════════════════════════════════════════╗
║              ANE Profiler Report                        ║
╚══════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────┐
│ Overall Metrics                                          │
├──────────────────────────────────────────────────────────┤
│ Total Steps:                  100                        │
│ Total Kernels:               1200                        │
│ Avg Step Time:              12.45 ms                     │
│ Total Compute Time:        1024.32 ms                    │
│ Total Transfer Time:         87.21 ms                    │
│ Total Compile Time:         156.00 ms                    │
│ ANE Utilization:             82.3 %                      │
│ Throughput:                 45.67 GB/s                   │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ Per-Kernel Statistics (sorted by time)                  │
├──────────────────────────────────────────────────────────┤
│ rmsnorm_layer_0                                         │
│   Calls:    100   Avg:     8.32 ms   Total:   832.0 ms ( 81.2%)  │
│   Throughput:  52.34 GB/s   Transfer:    12.3 ms                │
│   Min:     7.89 ms   Max:     9.12 ms                          │
│                                                          │
│ qkv_projection                                            │
│   Calls:    100   Avg:     2.45 ms   Total:   245.0 ms ( 23.9%)  │
│   Throughput:  38.21 GB/s   Transfer:     8.7 ms                │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ Bottleneck Analysis                                      │
├──────────────────────────────────────────────────────────┤
│ Slowest Kernel: rmsnorm_layer_0                          │
│   Takes 81.2% of total compute time                      │
│ ✓ Transfer overhead is acceptable                        │
│ ✓ Excellent ANE utilization: 82.3%                       │
└──────────────────────────────────────────────────────────┘
```

## Performance Guidelines

### ANE Utilization

| Utilization | Assessment | Action |
|-------------|------------|--------|
| > 80% | Excellent | No action needed |
| 50-80% | Good | Consider kernel fusion |
| < 50% | Low | Review pipelining, batch size |

### Transfer Overhead

| Ratio | Assessment | Action |
|-------|------------|--------|
| < 10% | Excellent | Optimal |
| 10-30% | Acceptable | Monitor |
| > 30% | High | Increase batch size, use async transfers |

### Throughput Expectations

| Operation | Expected Throughput (M4) |
|-----------|-------------------------|
| MatMul (large) | 50-100 GB/s |
| RMSNorm | 40-80 GB/s |
| Element-wise | 30-60 GB/s |
| Conv1x1 | 45-90 GB/s |

## Integration with Training Loop

```rust
use rustane::ane::ANEProfiler;

struct TrainingLoop {
    profiler: ANEProfiler,
    // ... other fields
}

impl TrainingLoop {
    fn new() -> Self {
        Self {
            profiler: ANEProfiler::new(),
            // ...
        }
    }

    fn train_step(&mut self, batch: Batch) -> Result<()> {
        self.profiler.start_step();

        // Forward pass
        let mut timer = KernelTimer::new(&mut self.profiler, "forward_rmsnorm");
        timer.with_data(batch.size() * self.dim * 2, batch.size() * self.dim * 2);
        self.forward_rmsnorm(&batch)?;
        // Timer drops automatically

        // ... more operations

        self.profiler.end_step();
        Ok(())
    }

    fn print_profile(&self) {
        println!("{}", self.profiler.generate_report());
    }
}
```

## Best Practices

1. **Enable profiling selectively** - Use `disabled()` constructor when not debugging
2. **Profile representative workloads** - Warm up before profiling
3. **Monitor trends** - Track metrics across many steps, not single measurements
4. **Identify bottlenecks** - Focus optimization on kernels with highest `pct_of_total`
5. **Watch transfer overhead** - High h2d/d2h time suggests batching opportunities

## Related Documentation

- `docs/ANE_CPU_ROUTING.md` - Operation routing decisions
- `docs/ANE_SIZE_CONSTRAINTS.md` - Size constraints and validation
- `docs/ANE_TRAINING_ARCHITECTURE.md` - Training system architecture
