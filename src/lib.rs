//! # Cairo Sierra to MLIR compiler and JIT engine

#![feature(arc_unwrap_or_clone)]
#![feature(box_into_inner)]
#![feature(error_generic_member_access)]
#![feature(hash_extract_if)]
#![feature(int_roundings)]
#![feature(iter_intersperse)]
#![feature(iterator_try_collect)]
#![feature(map_try_insert)]
#![feature(nonzero_ops)]
#![feature(provide_any)]
#![feature(strict_provenance)]
// #![warn(missing_docs)]
#![allow(clippy::missing_safety_doc)]

pub use self::{compiler::compile, jit_runner::execute};

mod compiler;
pub mod debug_info;
pub mod error;
mod ffi;
mod jit_runner;
pub mod libfuncs;
pub mod metadata;
pub mod types;
pub mod utils;
pub mod values;
