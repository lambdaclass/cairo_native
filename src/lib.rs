// # Cairo Sierra to MLIR compiler and JIT engine
//!
//! This crate is a compiler and JIT engine that transforms Sierra (or Cairo) sources into MLIR,
//! which can be [JIT-executed](https://en.wikipedia.org/wiki/Just-in-time_compilation) or further
//! compiled (externally) into a binary
//! [ahead of time](https://en.wikipedia.org/wiki/Ahead-of-time_compilation).
//!
//! ## Usage
//!
//! The API containts two structs, `NativeContext` and `NativeExecutor`.
//! The main purpose of `NativeContext` is MLIR initialization, compilation and lowering to LLVM.
//! `NativeExecutor` in the other hand is responsible of executing MLIR compiled sierra programs
//! from an entrypoint.
//! Programs and JIT states can be cached in contexts where their execution will be done multiple
//! times.
//!
//! ```
//! use cairo_native::context::NativeContext;
//! use cairo_native::executor::NativeExecutor;
//! use serde_json::json;
//! use std::{io::stdout, path::Path};
//!
//! // FIXME: Remove when cairo adds an easy to use API for setting the corelibs path.
//! std::env::set_var(
//!     "CARGO_MANIFEST_DIR",
//!     format!("{}/a", std::env::var("CARGO_MANIFEST_DIR").unwrap()),
//! );
//!
//! #[cfg(not(feature = "with-runtime"))]
//! compile_error!("This example requires the `with-runtime` feature to be active.");
//!
//! let name = cairo_native::utils::felt252_short_str("user");
//!
//! let program_path = Path::new("programs/examples/hello.cairo");

//!
//! // Compile the cairo program to sierra.
//! let sierra_program = cairo_native::utils::cairo_to_sierra(program_path);
//!
//! // Instantiate a Cairo Native MLIR contex. This data structure is responsible for the
//! // MLIR initialization and compilation of sierra programs into a MLIR module.
//! let native_context = NativeContext::new();
//!
//! // Compile the sierra program into a MLIR module. The MLIR program lowering into LLVM is done here too.
//! let native_program = native_context.compile(&sierra_program).expect("Compilation from Cairo to MLIR failed");
//!
//! // Print the resulting MLIR program.
//! println!(
//!     "{}",
//!     native_program
//!         .get_module()
//!         .as_operation()
//!         .to_string_with_flags(OperationPrintingFlags::new().enable_debug_info(true, false))
//!         .expect("MLIR printing failed")
//! );
//!
//! // At this point we could stop here, but let's continue with the JIT execution for the sake of
//! // completion.
//!
//! // Get necessary information for the execution of the program from a given entrypoint:
//! //   * entrypoint function id
//! //   * required initial gas
//! let entry_point = "hello::hello::greet";
//! let params = json!([name]);
//! let returns = &mut serde_json::Serializer::new(stdout());
//! let fn_id = cairo_native::utils::find_function_id(&sierra_program, entry_point);
//! let required_init_gas = native_program.get_required_init_gas(&fn_id);
//!
//! // JIT engine initialization. This engine may be cached in memory to keep potential JIT
//! // optimizations between invocations.
//! let native_executor = NativeExecutor::new(native_program);
//!
//! // Invoke the function.
//! native_executor.execute(&fn_id, params, returns, required_init_gas).unwrap();
//!
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

#![feature(alloc_layout_extra)]
#![feature(error_generic_member_access)]
#![feature(hash_extract_if)]
#![feature(nonzero_ops)]
#![feature(strict_provenance)]
// #![warn(missing_docs)]
#![allow(clippy::missing_safety_doc)]

pub use self::{compiler::compile, jit_runner::execute};

mod compiler;
pub mod context;
pub mod debug_info;
pub mod error;
pub mod executor;
mod ffi;
mod jit_runner;
pub mod libfuncs;
pub mod metadata;
pub mod module;
pub mod starknet;
pub mod types;
pub mod utils;
pub mod values;
