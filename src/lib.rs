//! [DeveloperDocumentation]: docs
#![doc = include_str!("../README.md")]
// #![warn(missing_docs)]
#![allow(clippy::missing_safety_doc)]

pub use self::{
    compiler::compile,
    ffi::{module_to_object, object_to_shared_lib, LLVMCompileError, OptLevel},
};

mod arch;
pub(crate) mod block_ext;
pub mod cache;
mod compiler;
pub mod context;
pub mod debug_info;
pub mod docs;
pub mod error;
pub mod execution_result;
pub mod executor;
mod ffi;
pub mod libfuncs;
pub mod metadata;
pub mod module;
pub mod starknet;
pub mod starknet_stub;
pub mod types;
pub mod utils;
pub mod values;
