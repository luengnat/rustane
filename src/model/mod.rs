//! GPT model architecture
//!
//! This module contains the GPT model implementation based on train_gpt.py

pub mod gpt;
pub mod gpt_model;

pub use gpt::{build_gpt_model, build_transformer_block, GptConfig, print_model_summary};
pub use gpt_model::{GptModel, CompiledGptModel};
