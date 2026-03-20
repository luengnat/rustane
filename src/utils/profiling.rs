//! Profiling utilities for memory usage and ANE utilization

use crate::layers::Shape;
use std::collections::HashMap;

/// Profile report containing profiling statistics
#[derive(Clone, Debug)]
pub struct ProfileReport {
    /// Host-side memory usage summary.
    pub memory_usage: MemoryStats,
    /// ANE compilation and execution counters.
    pub ane_stats: ANEStats,
}

impl std::fmt::Display for ProfileReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Profile Report:")?;
        writeln!(f)?;
        writeln!(f, "{}", self.memory_usage)?;
        writeln!(f)?;
        writeln!(f, "{}", self.ane_stats)
    }
}

/// Memory usage statistics
#[derive(Clone, Debug)]
pub struct MemoryStats {
    /// Total bytes tracked across all categories.
    pub total_bytes: usize,
    /// Bytes attributed to tensor storage.
    pub tensor_memory: usize,
    /// Bytes attributed to gradients.
    pub gradient_memory: usize,
    /// Bytes attributed to miscellaneous allocations.
    pub other_memory: usize,
}

impl std::fmt::Display for MemoryStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Memory Usage:")?;
        writeln!(f, "  Total: {} bytes", self.total_bytes)?;
        writeln!(f, "  Tensors: {} bytes", self.tensor_memory)?;
        writeln!(f, "  Gradients: {} bytes", self.gradient_memory)?;
        writeln!(f, "  Other: {} bytes", self.other_memory)
    }
}

/// ANE execution statistics
#[derive(Clone, Debug)]
pub struct ANEStats {
    /// Number of kernel compilations issued in-process.
    pub compile_count: usize,
    /// Number of ANE evaluations issued.
    pub execution_count: usize,
    /// Total model parameters associated with the profile.
    pub total_parameters: usize,
}

impl std::fmt::Display for ANEStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ANE Statistics:")?;
        writeln!(f, "  Compilations: {}", self.compile_count)?;
        writeln!(f, "  Executions: {}", self.execution_count)?;
        writeln!(f, "  Total Parameters: {}", self.total_parameters)
    }
}

/// Memory profiler for tracking tensor allocations
pub struct MemoryProfiler {
    tensor_memory: usize,
    gradient_memory: usize,
    other_memory: usize,
    allocations: HashMap<String, usize>,
}

impl MemoryProfiler {
    /// Create a new memory profiler
    pub fn new() -> Self {
        Self {
            tensor_memory: 0,
            gradient_memory: 0,
            other_memory: 0,
            allocations: HashMap::new(),
        }
    }

    /// Record tensor allocation
    ///
    /// # Arguments
    ///
    /// * `name` - Tensor name
    /// * `shape` - Tensor shape
    /// * `dtype` - Data type (FP32=4 bytes, FP16=2 bytes)
    pub fn record_tensor(&mut self, name: &str, shape: &Shape, dtype: DataType) {
        let bytes = shape.iter().product::<usize>() * dtype.bytes();

        self.tensor_memory += bytes;
        self.allocations.insert(name.to_string(), bytes);
    }

    /// Record gradient allocation
    pub fn record_gradient(&mut self, name: &str, shape: &Shape, dtype: DataType) {
        let bytes = shape.iter().product::<usize>() * dtype.bytes();

        self.gradient_memory += bytes;
        self.allocations.insert(format!("gradient_{}", name), bytes);
    }

    /// Record other allocation
    pub fn record_allocation(&mut self, name: &str, bytes: usize) {
        self.other_memory += bytes;
        self.allocations.insert(name.to_string(), bytes);
    }

    /// Get memory statistics
    pub fn get_stats(&self) -> MemoryStats {
        let total_bytes = self.tensor_memory + self.gradient_memory + self.other_memory;

        MemoryStats {
            total_bytes,
            tensor_memory: self.tensor_memory,
            gradient_memory: self.gradient_memory,
            other_memory: self.other_memory,
        }
    }

    /// Reset all tracking
    pub fn reset(&mut self) {
        self.tensor_memory = 0;
        self.gradient_memory = 0;
        self.other_memory = 0;
        self.allocations.clear();
    }
}

impl Default for MemoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// ANE profiler for tracking ANE usage
pub struct ANEProfiler {
    compile_count: usize,
    execution_count: usize,
    total_parameters: usize,
}

impl ANEProfiler {
    /// Create a new ANE profiler
    pub fn new() -> Self {
        Self {
            compile_count: 0,
            execution_count: 0,
            total_parameters: 0,
        }
    }

    /// Record a compilation
    pub fn record_compilation(&mut self) {
        self.compile_count += 1;
    }

    /// Record an execution
    pub fn record_execution(&mut self) {
        self.execution_count += 1;
    }

    /// Set total parameter count
    pub fn set_total_parameters(&mut self, count: usize) {
        self.total_parameters = count;
    }

    /// Get ANE statistics
    pub fn get_stats(&self) -> ANEStats {
        ANEStats {
            compile_count: self.compile_count,
            execution_count: self.execution_count,
            total_parameters: self.total_parameters,
        }
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        self.compile_count = 0;
        self.execution_count = 0;
    }
}

impl Default for ANEProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Data type for size calculation
#[derive(Clone, Copy, Debug)]
pub enum DataType {
    /// 32-bit floating point values.
    FP32,
    /// 16-bit floating point values.
    FP16,
    /// 8-bit signed integer values.
    INT8,
}

impl DataType {
    /// Get size in bytes
    pub fn bytes(&self) -> usize {
        match self {
            DataType::FP32 => 4,
            DataType::FP16 => 2,
            DataType::INT8 => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_profiler() {
        let mut profiler = MemoryProfiler::new();

        profiler.record_tensor("input", &vec![1, 256], DataType::FP32);
        profiler.record_tensor("weight", &vec![256, 512], DataType::FP32);
        profiler.record_gradient("grad", &vec![256, 512], DataType::FP32);

        let stats = profiler.get_stats();

        assert_eq!(stats.tensor_memory, 1 * 256 * 4 + 256 * 512 * 4);
        assert_eq!(stats.gradient_memory, 256 * 512 * 4);
        assert!(stats.total_bytes > 0);
    }

    #[test]
    fn test_ane_profiler() {
        let mut profiler = ANEProfiler::new();

        profiler.record_compilation();
        profiler.record_compilation();
        profiler.record_execution();
        profiler.set_total_parameters(1000);

        let stats = profiler.get_stats();

        assert_eq!(stats.compile_count, 2);
        assert_eq!(stats.execution_count, 1);
        assert_eq!(stats.total_parameters, 1000);
    }

    #[test]
    fn test_profile_report_display() {
        let report = ProfileReport {
            memory_usage: MemoryStats {
                total_bytes: 1000,
                tensor_memory: 800,
                gradient_memory: 150,
                other_memory: 50,
            },
            ane_stats: ANEStats {
                compile_count: 5,
                execution_count: 10,
                total_parameters: 1000,
            },
        };

        let display = format!("{}", report);
        assert!(display.contains("Memory Usage:"));
        assert!(display.contains("ANE Statistics:"));
    }
}
