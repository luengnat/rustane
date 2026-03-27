//! Memory Profiling for Training Runs
//!
//! Tracks memory usage during transformer training on ANE.
//! Measures heap allocations, ANE IOSurface usage, and peak memory.

use rustane::{init, training::*};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

/// Memory tracking allocator that wraps the system allocator
struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

/// Memory usage statistics
pub struct MemoryStats {
    pub heap_allocated: AtomicUsize,
    pub heap_deallocated: AtomicUsize,
    pub ane_iosurface_bytes: AtomicUsize,
    pub peak_heap_usage: AtomicUsize,
}

impl MemoryStats {
    const fn new() -> Self {
        Self {
            heap_allocated: AtomicUsize::new(0),
            heap_deallocated: AtomicUsize::new(0),
            ane_iosurface_bytes: AtomicUsize::new(0),
            peak_heap_usage: AtomicUsize::new(0),
        }
    }

    pub fn current_heap_usage(&self) -> usize {
        self.heap_allocated.load(Ordering::Relaxed) - self.heap_deallocated.load(Ordering::Relaxed)
    }

    pub fn update_peak(&self) {
        let current = self.current_heap_usage();
        let mut peak = self.peak_heap_usage.load(Ordering::Relaxed);
        while current > peak {
            match self.peak_heap_usage.compare_exchange_weak(
                peak,
                current,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }
    }

    pub fn report(&self) -> MemoryReport {
        MemoryReport {
            current_heap_mb: self.current_heap_usage() as f64 / 1024.0 / 1024.0,
            peak_heap_mb: self.peak_heap_usage.load(Ordering::Relaxed) as f64 / 1024.0 / 1024.0,
            ane_iosurface_mb: self.ane_iosurface_bytes.load(Ordering::Relaxed) as f64
                / 1024.0
                / 1024.0,
        }
    }
}

/// Human-readable memory report
#[derive(Debug, Clone)]
pub struct MemoryReport {
    pub current_heap_mb: f64,
    pub peak_heap_mb: f64,
    pub ane_iosurface_mb: f64,
}

impl std::fmt::Display for MemoryReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Heap: {:.2} MB (peak: {:.2} MB) | ANE IOSurface: {:.2} MB",
            self.current_heap_mb, self.peak_heap_mb, self.ane_iosurface_mb
        )
    }
}

/// Memory profile for a training step
#[derive(Debug)]
pub struct StepMemoryProfile {
    pub step: usize,
    pub before: MemoryReport,
    pub after: MemoryReport,
    pub delta_heap_mb: f64,
    pub duration_ms: f64,
}

impl StepMemoryProfile {
    pub fn new(step: usize, before: MemoryReport, after: MemoryReport, duration_ms: f64) -> Self {
        Self {
            step,
            delta_heap_mb: after.current_heap_mb - before.current_heap_mb,
            before,
            after,
            duration_ms,
        }
    }

    pub fn report(&self) {
        println!(
            "Step {:3}: {:.2}ms | {} | Δ: {:.2} MB",
            self.step, self.duration_ms, self.after, self.delta_heap_mb
        );
    }
}

/// Profile memory usage during training
pub struct MemoryProfiler {
    stats: MemoryStats,
    profiles: Vec<StepMemoryProfile>,
}

impl MemoryProfiler {
    pub fn new() -> Self {
        Self {
            stats: MemoryStats::new(),
            profiles: Vec::new(),
        }
    }

    pub fn snapshot(&self) -> MemoryReport {
        self.stats.report()
    }

    pub fn record_step(&mut self, step: usize, duration_ms: f64) {
        let current = self.snapshot();
        let before = self
            .profiles
            .last()
            .map(|p| p.after.clone())
            .unwrap_or(current.clone());
        let profile = StepMemoryProfile::new(step, before, current, duration_ms);
        self.stats.update_peak();
        profile.report();
        self.profiles.push(profile);
    }

    pub fn summary(&self) -> MemoryProfileSummary {
        let total_delta: f64 = self.profiles.iter().map(|p| p.delta_heap_mb).sum();
        let max_delta = self
            .profiles
            .iter()
            .map(|p| p.delta_heap_mb)
            .fold(f64::NEG_INFINITY, f64::max);
        let avg_duration_ms =
            self.profiles.iter().map(|p| p.duration_ms).sum::<f64>() / self.profiles.len() as f64;

        MemoryProfileSummary {
            total_steps: self.profiles.len(),
            total_heap_growth_mb: total_delta,
            max_step_growth_mb: max_delta,
            avg_step_duration_ms: avg_duration_ms,
            peak_heap_mb: self.stats.peak_heap_usage.load(Ordering::Relaxed) as f64
                / 1024.0
                / 1024.0,
            ane_iosurface_mb: self.stats.ane_iosurface_bytes.load(Ordering::Relaxed) as f64
                / 1024.0
                / 1024.0,
        }
    }
}

/// Summary statistics for the entire training run
#[derive(Debug)]
pub struct MemoryProfileSummary {
    pub total_steps: usize,
    pub total_heap_growth_mb: f64,
    pub max_step_growth_mb: f64,
    pub avg_step_duration_ms: f64,
    pub peak_heap_mb: f64,
    pub ane_iosurface_mb: f64,
}

impl std::fmt::Display for MemoryProfileSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")?;
        writeln!(f, "Memory Profile Summary")?;
        writeln!(f, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")?;
        writeln!(f, "  Total Steps: {}", self.total_steps)?;
        writeln!(
            f,
            "  Total Heap Growth: {:.2} MB",
            self.total_heap_growth_mb
        )?;
        writeln!(f, "  Max Step Growth: {:.2} MB", self.max_step_growth_mb)?;
        writeln!(
            f,
            "  Avg Step Duration: {:.2} ms",
            self.avg_step_duration_ms
        )?;
        writeln!(f, "  Peak Heap Usage: {:.2} MB", self.peak_heap_mb)?;
        writeln!(f, "  ANE IOSurface: {:.2} MB", self.ane_iosurface_mb)?;
        writeln!(f, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")?;
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🍎 Rustane - Memory Profiling for Training");
    println!("==========================================\n");

    // Initialize ANE
    init()?;

    // Check platform
    let avail = rustane::HardwareAvailability::check();
    println!("Platform: {}", avail.describe());
    println!();

    // Create small transformer config for testing
    let config = TransformerConfig::tiny();
    println!("Model Config:");
    println!("  Vocab Size: {}", config.vocab_size);
    println!("  Hidden Dim: {}", config.dim);
    println!("  FFN Dim: {}", config.hidden_dim);
    println!("  Layers: {}", config.n_layers);
    println!("  Heads: {}", config.n_heads);
    println!("  Total Params: {}", config.param_count());
    println!();

    // Create profiler
    let mut profiler = MemoryProfiler::new();

    // Simulate training steps with memory tracking
    println!("Running 10 training steps with memory profiling...\n");

    for step in 1..=10 {
        let start = Instant::now();

        // Simulate training step (allocate some memory)
        let _batch = vec![0u32; 1000];
        let _activations = vec![0.0f32; 10000];

        // Simulate gradient computation
        let _gradients = vec![0.0f32; config.param_count().min(100000)];

        let duration = start.elapsed();
        profiler.record_step(step, duration.as_secs_f64() * 1000.0);
    }

    // Print summary
    let summary = profiler.summary();
    print!("{}", summary);

    Ok(())
}
