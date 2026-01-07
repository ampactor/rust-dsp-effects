//! # rust-dsp-effects
//!
//! High-performance, embedded-ready audio effects framework.
//!
//! This crate provides a collection of professional-grade audio effects
//! designed for real-time processing with zero heap allocations in the
//! audio path.
//!
//! ## Design Goals
//!
//! - **`no_std` compatible**: Runs on embedded systems with ~512KB RAM
//! - **Zero-allocation processing**: All buffers pre-allocated, no heap in audio path
//! - **Real-time safe**: Deterministic execution, no locks in process calls
//! - **Professional quality**: Proper DSP algorithms, not shortcuts
//!
//! ## Quick Start
//!
//! ```rust
//! use rust_dsp_effects::{Effect, SmoothedParam};
//!
//! // Effects implement the Effect trait
//! // Process single samples or blocks
//! // Parameters use SmoothedParam for zipper-free changes
//! ```

// Enable no_std when std feature is disabled
#![cfg_attr(not(feature = "std"), no_std)]

pub mod effect;
pub mod param;

// Re-export main types at crate root
pub use effect::{Chain, Effect, EffectExt};
pub use param::{LinearSmoothedParam, SmoothedParam};
