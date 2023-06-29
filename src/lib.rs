//! # Cairo Sierra to MLIR compiler and JIT engine

#![feature(arc_unwrap_or_clone)]
#![feature(box_into_inner)]
#![feature(int_roundings)]
#![feature(iter_intersperse)]
#![feature(iterator_try_collect)]
#![feature(map_try_insert)]
#![feature(nonzero_ops)]
#![feature(strict_provenance)]
// #![warn(missing_docs)]
#![allow(clippy::missing_safety_doc)]

pub use self::{compiler::compile, debug_info::DebugInfo, jit_runner::execute};

mod compiler;
mod debug_info;
mod ffi;
mod jit_runner;
pub mod libfuncs;
pub mod metadata;
pub mod types;
pub mod utils;
pub mod values;
