//! ANE Tiling - Split large operations into ANE-compatible chunks
//!
//! ANE has a hard limit of 16,384 elements per tensor.
//! This module implements tiling strategies to split larger operations.

use super::mil_generator::{ANEShape, ANETensorType};
use super::training_architecture::KernelTemplate;

/// Tile configuration for ANE operations
#[derive(Debug, Clone)]
pub struct TileConfig {
    /// Number of tiles along channel dimension
    pub n_channel_tiles: usize,
    /// Number of tiles along spatial dimension  
    pub n_spatial_tiles: usize,
    /// Elements per tile (must be <= 16384)
    pub elements_per_tile: usize,
    /// Whether tiling is needed
    pub needs_tiling: bool,
}

impl TileConfig {
    /// Calculate optimal tiling for a shape
    ///
    /// Strategy:
    /// 1. Try to fit entire operation in one tile
    /// 2. If too large, split along spatial dimension first (seq_len)
    /// 3. If still too large, split along channel dimension
    pub fn for_shape(shape: &ANEShape) -> Self {
        let total_elements = shape.num_elements();
        let max_elements = 16384;

        if total_elements <= max_elements {
            return Self {
                n_channel_tiles: 1,
                n_spatial_tiles: 1,
                elements_per_tile: total_elements,
                needs_tiling: false,
            };
        }

        // Need tiling - try spatial first (usually seq_len)
        let spatial = shape.w;
        let channels = shape.c;

        // Calculate how many spatial tiles needed
        let elements_per_spatial_slice = channels;
        let max_spatial_per_tile = max_elements / elements_per_spatial_slice;

        if max_spatial_per_tile >= 1 {
            let n_spatial_tiles = (spatial + max_spatial_per_tile - 1) / max_spatial_per_tile;
            let spatial_per_tile = spatial / n_spatial_tiles;

            return Self {
                n_channel_tiles: 1,
                n_spatial_tiles,
                elements_per_tile: channels * spatial_per_tile,
                needs_tiling: true,
            };
        }

        // Need to tile channels too
        let n_channel_tiles = (channels + max_elements - 1) / max_elements;
        let channels_per_tile = channels / n_channel_tiles;

        Self {
            n_channel_tiles,
            n_spatial_tiles: 1,
            elements_per_tile: channels_per_tile * spatial,
            needs_tiling: true,
        }
    }

    /// Get tile boundaries for a given tile index
    pub fn get_tile_bounds(
        &self,
        tile_idx: usize,
        total_channels: usize,
        total_spatial: usize,
    ) -> (usize, usize, usize, usize) {
        // Returns (c_start, c_end, s_start, s_end)
        let channels_per_tile = total_channels / self.n_channel_tiles;
        let spatial_per_tile = total_spatial / self.n_spatial_tiles;

        let channel_tile = tile_idx / self.n_spatial_tiles;
        let spatial_tile = tile_idx % self.n_spatial_tiles;

        let c_start = channel_tile * channels_per_tile;
        let c_end = if channel_tile == self.n_channel_tiles - 1 {
            total_channels
        } else {
            (channel_tile + 1) * channels_per_tile
        };

        let s_start = spatial_tile * spatial_per_tile;
        let s_end = if spatial_tile == self.n_spatial_tiles - 1 {
            total_spatial
        } else {
            (spatial_tile + 1) * spatial_per_tile
        };

        (c_start, c_end, s_start, s_end)
    }
}

/// Tiled kernel generator
///
/// Creates multiple small kernels that together perform one large operation
pub struct TiledKernel {
    /// Original kernel template
    pub template: KernelTemplate,
    /// Tile configuration
    pub tile_config: TileConfig,
    /// Individual tile kernels
    pub tile_kernels: Vec<KernelTemplate>,
}

impl TiledKernel {
    /// Create a tiled version of a kernel
    pub fn new(template: KernelTemplate) -> Self {
        let (channels, spatial) = match &template {
            KernelTemplate::RmsNorm { channels, seq_len } => (*channels, *seq_len),
            KernelTemplate::DynamicLinear {
                in_features,
                seq_len,
                ..
            } => (*in_features, *seq_len),
            KernelTemplate::QkvProjection { dim, seq_len, .. } => (*dim, *seq_len),
            KernelTemplate::SdpaForward {
                heads,
                head_dim,
                seq_len,
            } => (heads * head_dim, *seq_len),
            KernelTemplate::FfnSwiglu { dim, seq_len, .. } => (*dim, *seq_len),
        };

        let shape = ANEShape::seq(channels, spatial);
        let tile_config = TileConfig::for_shape(&shape);

        let mut tile_kernels = Vec::new();

        if tile_config.needs_tiling {
            // Generate tile kernels
            let total_tiles = tile_config.n_channel_tiles * tile_config.n_spatial_tiles;

            for tile_idx in 0..total_tiles {
                let (c_start, c_end, s_start, s_end) =
                    tile_config.get_tile_bounds(tile_idx, channels, spatial);

                let tile_channels = c_end - c_start;
                let tile_spatial = s_end - s_start;

                // Create tile-specific kernel
                let tile_kernel = Self::create_tile_kernel(
                    &template,
                    tile_idx,
                    tile_channels,
                    tile_spatial,
                    c_start,
                    s_start,
                );

                tile_kernels.push(tile_kernel);
            }
        } else {
            // No tiling needed - use original kernel
            tile_kernels.push(template.clone());
        }

        Self {
            template,
            tile_config,
            tile_kernels,
        }
    }

    /// Create a kernel for a specific tile
    fn create_tile_kernel(
        original: &KernelTemplate,
        tile_idx: usize,
        tile_channels: usize,
        tile_spatial: usize,
        channel_offset: usize,
        spatial_offset: usize,
    ) -> KernelTemplate {
        // For now, create simplified versions
        // In full implementation, would generate MIL with slice operations
        match original {
            KernelTemplate::RmsNorm { .. } => KernelTemplate::RmsNorm {
                channels: tile_channels,
                seq_len: tile_spatial,
            },
            KernelTemplate::DynamicLinear { out_features, .. } => KernelTemplate::DynamicLinear {
                in_features: tile_channels,
                out_features: *out_features,
                seq_len: tile_spatial,
            },
            KernelTemplate::QkvProjection { q_dim, kv_dim, .. } => KernelTemplate::QkvProjection {
                dim: tile_channels,
                q_dim: *q_dim,
                kv_dim: *kv_dim,
                seq_len: tile_spatial,
            },
            _ => original.clone(), // Fallback for unsupported types
        }
    }

    /// Get total number of tiles
    pub fn n_tiles(&self) -> usize {
        self.tile_kernels.len()
    }

    /// Check if tiling was needed
    pub fn is_tiled(&self) -> bool {
        self.tile_config.needs_tiling
    }
}

/// Tiling-aware training configuration
///
/// Automatically generates tiled kernels for operations that exceed ANE limits
pub struct TiledTrainingConfig {
    /// Base configuration
    pub base_config: super::training_architecture::ANETrainingConfig,
    /// Tiled kernels for each operation
    pub tiled_kernels: Vec<TiledKernel>,
    /// Total kernel count (including tiles)
    pub total_kernel_count: usize,
}

impl TiledTrainingConfig {
    /// Create tiled configuration from base config
    pub fn from_config(config: super::training_architecture::ANETrainingConfig) -> Self {
        let base_kernels = config.generate_kernels();
        let mut tiled_kernels = Vec::new();
        let mut total_count = 0;

        for kernel in base_kernels {
            let tiled = TiledKernel::new(kernel);
            total_count += tiled.n_tiles();
            tiled_kernels.push(tiled);
        }

        Self {
            base_config: config,
            tiled_kernels,
            total_kernel_count: total_count,
        }
    }

    /// Get all individual tile kernels flattened
    pub fn get_all_kernels(&self) -> Vec<KernelTemplate> {
        self.tiled_kernels
            .iter()
            .flat_map(|tk| tk.tile_kernels.clone())
            .collect()
    }

    /// Check if config fits within compile budget
    pub fn fits_budget(&self, budget: &super::training_architecture::CompileBudget) -> bool {
        budget.request_compile(self.total_kernel_count as i32)
    }

    /// Print tiling report
    pub fn print_report(&self) {
        println!("\n=== Tiling Report ===\n");
        println!("Base kernels: {}", self.tiled_kernels.len());
        println!("Total tiles: {}", self.total_kernel_count);
        println!("\nBreakdown:");

        for (i, tiled) in self.tiled_kernels.iter().enumerate() {
            let status = if tiled.is_tiled() {
                format!("{} tiles", tiled.n_tiles())
            } else {
                "no tiling".to_string()
            };

            println!("  {}. {} - {}", i + 1, tiled.template.id(), status);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_tiling_needed() {
        let shape = ANEShape::seq(64, 64); // 4096 elements
        let config = TileConfig::for_shape(&shape);

        assert!(!config.needs_tiling);
        assert_eq!(config.n_channel_tiles, 1);
        assert_eq!(config.n_spatial_tiles, 1);
    }

    #[test]
    fn test_spatial_tiling() {
        let shape = ANEShape::seq(512, 512); // 262K elements, need tiling
        let config = TileConfig::for_shape(&shape);

        assert!(config.needs_tiling);
        assert_eq!(config.n_channel_tiles, 1);
        assert!(config.n_spatial_tiles > 1);
        assert!(config.elements_per_tile <= 16384);
    }

    #[test]
    fn test_tile_bounds() {
        let config = TileConfig {
            n_channel_tiles: 2,
            n_spatial_tiles: 2,
            elements_per_tile: 4096,
            needs_tiling: true,
        };

        let (c_start, c_end, s_start, s_end) = config.get_tile_bounds(0, 128, 128);

        assert_eq!(c_start, 0);
        assert_eq!(c_end, 64);
        assert_eq!(s_start, 0);
        assert_eq!(s_end, 64);
    }
}
