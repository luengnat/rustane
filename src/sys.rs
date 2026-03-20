//! Raw ANE low-level bindings compatibility layer.
//!
//! The implementation now lives in [`crate::ane`], but we keep this module so
//! the existing wrapper layer can continue importing `crate::sys::*`.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

pub(crate) use crate::ane::sys::*;

#[cfg(test)]
mod tests {
    #[test]
    fn test_bindings_compile() {}
}
