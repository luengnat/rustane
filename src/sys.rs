// Raw FFI bindings to ANE bridge C API
//
// This module contains unsafe bindings to the ANE bridge library,
// which wraps Apple's private ANE APIs (_ANEClient, _ANECompiler).
//
// # Safety
//
// All functions in this module are unsafe and require careful handling:
// - ANEKernelHandle must be properly managed (created via compile, freed via ane_bridge_free)
// - Tensor I/O functions require correct buffer sizes matching the compiled kernel's expectations
// - The ANE runtime must be initialized via ane_bridge_init before any other operations
//
// For safe, idiomatic Rust wrappers, see the parent crate modules.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    #[test]
    fn test_bindings_compile() {
        // This test just verifies that the bindings compile successfully
        // Actual ANE operations require Apple Silicon hardware
    }
}
