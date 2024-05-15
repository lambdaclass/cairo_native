//! # Cairo Sierra to MLIR compiler and JIT engine
//!
//! This crate is a compiler and JIT engine that transforms Sierra (or Cairo) sources into MLIR,
//! which can be [JIT-executed](https://en.wikipedia.org/wiki/Just-in-time_compilation) or further
//! compiled (externally) into a binary
//! [ahead of time](https://en.wikipedia.org/wiki/Ahead-of-time_compilation).
//!
//! ## Usage
//!
//! The API contains two structs, `NativeContext` and `NativeExecutor`.
//! The main purpose of `NativeContext` is MLIR initialization, compilation and lowering to LLVM.
//! `NativeExecutor` in the other hand is responsible of executing MLIR compiled sierra programs
//! from an entrypoint.
//! Programs and JIT states can be cached in contexts where their execution will be done multiple
//! times.
//!
//! ```
//! use starknet_types_core::felt::Felt;
//! use cairo_native::context::NativeContext;
//! use cairo_native::executor::JitNativeExecutor;
//! use cairo_native::values::JitValue;
//! use std::path::Path;
//!
//! let program_path = Path::new("programs/examples/hello.cairo");
//! // Compile the cairo program to sierra.
//! let sierra_program = cairo_native::utils::cairo_to_sierra(program_path);
//!
//! // Instantiate a Cairo Native MLIR context. This data structure is responsible for the MLIR
//! // initialization and compilation of sierra programs into a MLIR module.
//! let native_context = NativeContext::new();
//!
//! // Compile the sierra program into a MLIR module.
//! let native_program = native_context.compile(&sierra_program, None).unwrap();
//!
//! // The parameters of the entry point.
//! let params = &[JitValue::Felt252(Felt::from_bytes_be_slice(b"user"))];
//!
//! // Find the entry point id by its name.
//! let entry_point = "hello::hello::greet";
//! let entry_point_id = cairo_native::utils::find_function_id(&sierra_program, entry_point);
//!
//! // Instantiate the executor.
//! let native_executor = JitNativeExecutor::from_native_module(
//!     native_context.context(),
//!     native_program,
//!     Default::default()
//! );
//!
//! // Execute the program.
//! let result = native_executor
//!     .invoke_dynamic(entry_point_id, params, None)
//!     .unwrap();
//!
//! println!("Cairo program was compiled and executed successfully.");
//! println!("{:?}", result);
//! ```
//!
//! ## Common definitions
//!
//! Within this project there are lots of functions with the same signature. As their arguments have
//! all the same meaning, they are documented here:
//!
//!   - `context: NativeContext`: The MLIR context.
//!   - `module: &NativeModule`: The compiled MLIR program, with other relevant information such as program registry and metadata.
//!   - `program: &Program`: The Sierra input program.
//!   - `registry: &ProgramRegistry<TType, TLibfunc>`: The registry extracted from the program.
//!   - `metadata: &mut MetadataStorage`: Current compiler metadata.

// #![warn(missing_docs)]
#![allow(clippy::missing_safety_doc)]

pub use self::{
    compiler::compile,
    ffi::{module_to_object, object_to_shared_lib, LLVMCompileError, OptLevel},
};

pub(crate) mod block_ext;
pub mod cache;
mod compiler;
pub mod context;
pub mod debug_info;
pub mod error;
pub mod execution_result;
pub mod executor;
mod ffi;
pub mod libfuncs;
pub mod metadata;
pub mod module;
pub mod starknet;
pub mod types;
pub mod utils;
pub mod values;
