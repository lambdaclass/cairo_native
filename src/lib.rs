//! # Cairo Sierra to MLIR compiler and JIT engine
//!
//! This crate is a compiler and JIT engine that transforms Sierra (or Cairo) sources into MLIR,
//! which can then be executed using MLIR's
//! [JIT](https://en.wikipedia.org/wiki/Just-in-time_compilation) engine, or compiled (externally)
//! into a binary or a library ([AOT](https://en.wikipedia.org/wiki/Ahead-of-time_compilation)).
//!
//! ## Usage
//!
//! The API contains two types for interfacing: `NativeContext` and `NativeExecutor`. The main
//! purpose of `NativeContext` is MLIR initialization, compilation and lowering to LLVM.
//! `NativeExecutor` is responsible of executing MLIR compiled sierra programs starting from an
//! entrypoint, using either the JIT or an AOT-compiled library. Programs (both AOT and
//! JIT-compiled) can be cached to be reused later on.
//!
//! ```
//! # use starknet_types_core::felt::Felt;
//! # use cairo_native::{
//! #     context::NativeContext,
//! #     executor::JitNativeExecutor,
//! #     values::JitValue,
//! # };
//! # use std::path::Path;
//! #
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // Compile the cairo program to sierra.
//! let sierra_program = cairo_native::utils::cairo_to_sierra("programs/examples/hello.cairo");
//!
//! // Create a new Cairo compilation context and use it to compile our Cairo program.
//! let native_context = NativeContext::new();
//! let native_program = native_context.compile(&sierra_program)?;
//!
//! // We want to call the `hello::hello::greet` function with an argument containing the
//! // representation of the string `user` as a felt.
//! let entry_point = "hello::hello::greet";
//! let arguments = &[JitValue::Felt252(Felt::from_bytes_be_slice(b"user"))];
//!
//! // To execute the program we need an engine. In this example we'll use the JIT engine.
//! let native_executor = JitNativeExecutor::from_native_module(native_program, Default::default());
//!
//! // Obtain the Sierra function ID of the entry point and execute it using the engine.
//! let entry_point_id = cairo_native::utils::find_function_id(&sierra_program, entry_point);
//! let result = native_executor.invoke_dynamic(entry_point_id, arguments, None, None)?;
//!
//! println!("Cairo program was compiled and executed successfully.");
//! println!("{:?}", result);
//! #
//! # Ok(())
//! # }
//! #
//! # run().unwrap();
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

#![warn(missing_docs)]
#![allow(clippy::missing_safety_doc)]

pub use self::{
    compiler::compile,
    ffi::{module_to_object, object_to_shared_lib, LLVMCompileError, OptLevel},
};

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
