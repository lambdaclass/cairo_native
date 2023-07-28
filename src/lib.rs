//! # Cairo Sierra to MLIR compiler and JIT engine
//!
//! This crate is a compiler and JIT engine that transforms Sierra (or Cairo) sources into MLIR,
//! which can be [JIT-executed](https://en.wikipedia.org/wiki/Just-in-time_compilation) or further
//! compiled (externally) into a binary
//! [ahead of time](https://en.wikipedia.org/wiki/Ahead-of-time_compilation).
//!
//! ## Usage
//!
//! Right now, there are two ways of using this crate: the easy API and the normal one.
//!
//! **Easy API**
//!
//! The [easy API](crate::easy) contains a single function,
//! [`compile_and_execute`](crate::easy::compile_and_execute) which combines both functions in the
//! normal API as well as all the required initialization and customization. It may be used for
//! testing and prototyping, but it's **NOT intended** to be anything more than that.
//!
//! ```
//! # use cairo_lang_compiler::CompilerConfig;
//! # use cairo_lang_sierra::extensions::core::{CoreLibfunc, CoreType};
//! # use num_bigint::BigUint;
//! # use serde_json::json;
//! # use std::{io::stdout, path::Path};
//! #
//! # // FIXME: Remove when cairo adds an easy to use API for setting the corelibs path.
//! # std::env::set_var(
//! #     "CARGO_MANIFEST_DIR",
//! #     format!("{}/a", std::env::var("CARGO_MANIFEST_DIR").unwrap()),
//! # );
//! #
//! # #[cfg(not(feature = "with-runtime"))]
//! # compile_error!("This example requires the `with-runtime` feature to be active.");
//! #
//! # let program = cairo_lang_compiler::compile_cairo_project_at_path(
//! #     Path::new("programs/examples/hello.cairo"),
//! #     CompilerConfig {
//! #         replace_ids: true,
//! #         ..Default::default()
//! #     },
//! # )
//! # .unwrap();
//! #
//! # let name = {
//! #     let mut digits = BigUint::from(u32::from_le_bytes(*b"user")).to_u32_digits();
//! #     digits.resize(8, 0);
//! #     digits
//! # };
//! #
//! // The easy API requires only the program, the entry point and the de/serializers.
//! cairo_native::easy::compile_and_execute::<CoreType, CoreLibfunc, _, _>(
//!     &program,
//!     &program
//!         .funcs
//!         .iter()
//!         .find(|x| x.id.debug_name.as_deref() == Some("hello::hello::greet"))
//!         .unwrap()
//!         .id,
//!     json!([[1919251317, 0, 0, 0, 0, 0, 0, 0]]),
//!     &mut serde_json::Serializer::new(stdout()),
//! )
//! .unwrap();
//! # println!();
//! ```
//!
//! **Normal API**
//!
//! The normal API contains two different functions: [`compile`] and [`execute`]. They require some
//! initialization, but are much more powerful since they allow, for example, the caching of
//! programs and JIT states.
//!
//! ```
//! # use cairo_lang_compiler::CompilerConfig;
//! # use cairo_lang_sierra::{
//! #     extensions::core::{CoreLibfunc, CoreType},
//! #     program_registry::ProgramRegistry,
//! # };
//! # use cairo_native::metadata::{runtime_bindings::RuntimeBindingsMeta, MetadataStorage};
//! # use melior::{
//! #     dialect::DialectRegistry,
//! #     ir::{operation::OperationPrintingFlags, Location, Module},
//! #     pass::{self, PassManager},
//! #     utility::register_all_dialects,
//! #     Context,
//! #     ExecutionEngine,
//! # };
//! # use serde_json::json;
//! # use std::{io::stdout, path::Path};
//! #
//! # // FIXME: Remove when cairo adds an easy to use API for setting the corelibs path.
//! # std::env::set_var(
//! #     "CARGO_MANIFEST_DIR",
//! #     format!("{}/a", std::env::var("CARGO_MANIFEST_DIR").unwrap()),
//! # );
//! #
//! # #[cfg(not(feature = "with-runtime"))]
//! # compile_error!("This example requires the `with-runtime` feature to be active.");
//! #
//! # // Load Sierra program and its registry.
//! # let program = cairo_lang_compiler::compile_cairo_project_at_path(
//! #     Path::new("programs/examples/hello.cairo"),
//! #     CompilerConfig {
//! #         replace_ids: true,
//! #         ..Default::default()
//! #     },
//! # )
//! # .unwrap();
//! # let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program)
//! #     .expect("Failed to initialize the Sierra program's registry");
//! #
//! // MLIR initialization.
//! let context = Context::new();
//! context.append_dialect_registry(&{
//!     let registry = DialectRegistry::new();
//!     register_all_dialects(&registry);
//!     registry
//! });
//! context.load_all_available_dialects();
//!
//! // Module to store the compiled program in.
//! let mut module = Module::new(Location::unknown(&context));
//!
//! // Metadata initialization.
//! let mut metadata = MetadataStorage::new();
//! metadata.insert(RuntimeBindingsMeta::default()).unwrap(); // Only if using the runtime library.
//!
//! // Actual compilation. The compiled program will be stored in `module`.
//! // If the program was already compiled and stored, the `parse` method of `melior::ir::Module`
//! // will parse it.
//! cairo_native::compile::<CoreType, CoreLibfunc>(
//!     &context,
//!     &module,
//!     &program,
//!     &registry,
//!     &mut metadata,
//!     None, // If provided, some operations will contain source locations.
//! )
//! .expect("Compilation from Cairo to Sierra failed");
//!
//! // Print the resulting MLIR program.
//! println!(
//!     "{}",
//!     module
//!         .as_operation()
//!         .to_string_with_flags(OperationPrintingFlags::new().enable_debug_info(true, false))
//!         .expect("MLIR printing failed")
//! );
//!
//! // At this point we could stop here, but let's continue with the JIT section for the sake of
//! // completion.
//!
//! // MLIR program lowering into LLVM.
//! let pass_manager = PassManager::new(&context);
//! pass_manager.enable_verifier(true);
//! pass_manager.add_pass(pass::transform::create_canonicalizer());
//! pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
//! pass_manager.add_pass(pass::conversion::create_arith_to_llvm());
//! pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
//! pass_manager.add_pass(pass::conversion::create_func_to_llvm());
//! pass_manager.add_pass(pass::conversion::create_index_to_llvm_pass());
//! pass_manager.add_pass(pass::conversion::create_mem_ref_to_llvm());
//! pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());
//! pass_manager.run(&mut module).expect("MLIR lowering passes failed");
//!
//! // JIT engine initialization. This engine may be cached in memory to keep potential JIT
//! // optimizations between invocations.
//! let engine = ExecutionEngine::new(&module, 3, &[], false);
//! cairo_native::utils::register_runtime_symbols(&engine); // Only if using the runtime library.
//!
//! // Invoke the function.
//! cairo_native::execute::<CoreType, CoreLibfunc, _, _>(
//!     &engine,
//!     &registry,
//!     &program
//!         .funcs
//!         .iter()
//!         .find(|x| x.id.debug_name.as_deref() == Some("hello::hello::greet"))
//!         .unwrap()
//!         .id,
//!     json!([[1919251317, 0, 0, 0, 0, 0, 0, 0]]),
//!     &mut serde_json::Serializer::pretty(stdout()),
//! )
//! .unwrap();
//! ```
//!
//! ## Common definitions
//!
//! Within this project there are lots of functions with the same signature. As their arguments have
//! all the same meaning, they are documented here:
//!
//!   - `context: Context`: The MLIR context.
//!   - `module: &Module`: The compiled MLIR program.
//!   - `program: &Program`: The Sierra input program.
//!   - `registry: &ProgramRegsitry<TType, TLibfunc>`: The registry extracted from the program.
//!   - `metadata: &mut MetadataStorage`: Current compiler metadata.

#![feature(alloc_layout_extra)]
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
pub mod easy;
pub mod error;
mod ffi;
mod jit_runner;
pub mod libfuncs;
pub mod metadata;
pub mod starknet;
pub mod types;
pub mod utils;
pub mod values;
