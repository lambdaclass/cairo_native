// # Cairo Sierra to MLIR compiler and JIT engine
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
//! let name = cairo_native::easy::felt252_short_str("user");
//!
//! // The easy API requires only the program, the entry point and the de/serializers.
//! cairo_native::easy::compile_and_execute(
//!     Path::new("programs/examples/hello.cairo"),
//!     "hello::hello::greet",
//!     json!([name]),
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
//!   - `registry: &ProgramRegistry<TType, TLibfunc>`: The registry extracted from the program.
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
#![feature(strict_provenance)]
// #![warn(missing_docs)]
#![allow(clippy::missing_safety_doc)]

pub use self::{compiler::compile, jit_runner::execute};
use crate::error::JitRunnerError;
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    ids::FunctionId,
    program::Program,
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::DialectRegistry,
    ir::{Location, Module},
    pass::{self, PassManager},
    utility::{register_all_dialects, register_all_passes},
    Context, ExecutionEngine,
};
use metadata::{
    gas::{GasMetadata, MetadataComputationConfig},
    runtime_bindings::RuntimeBindingsMeta,
    MetadataStorage,
};
use serde::{Deserializer, Serializer};
use std::any::Any;
use utils::create_engine;

mod compiler;
pub mod debug_info;
pub mod error;
mod ffi;
mod jit_runner;
pub mod libfuncs;
pub mod metadata;
pub mod starknet;
pub mod types;
pub mod utils;
pub mod values;

/// Context of IRs, dialects and passes for Cairo programs compilation.
pub struct NativeContext {
    context: Context,
}

impl NativeContext {
    pub fn new() -> Self {
        let context = initialize_mlir();
        Self { context }
    }

    pub fn compile(
        &self,
        program: &Program,
    ) -> Result<NativeModule, error::CompileError<CoreType, CoreLibfunc>> {
        let mut module = Module::new(Location::unknown(&self.context));

        let has_gas_builtin = program
            .type_declarations
            .iter()
            .any(|decl| decl.long_id.generic_id.0.as_str() == "GasBuiltin");

        let mut metadata = MetadataStorage::new();

        // Make the runtime library available.
        metadata.insert(RuntimeBindingsMeta::default());
        // We assume that GasMetadata will be always present when the program uses the gas builtin.
        if has_gas_builtin {
            let gas_metadata = GasMetadata::new(program, MetadataComputationConfig::default());
            // Unwrapping here is not necessary since the insertion will only fail if there was
            // already some metadata of the same type.
            metadata.insert(gas_metadata);
        }

        // Create the Sierra program registry
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(program)?;

        crate::compile(
            &self.context,
            &module,
            program,
            &registry,
            &mut metadata,
            None,
        )?;

        self.lower_to_llvm(&mut module)?;

        Ok(NativeModule::new(module, registry, metadata))
    }

    fn lower_to_llvm(
        &self,
        module: &mut Module,
    ) -> Result<(), error::CompileError<CoreType, CoreLibfunc>> {
        let pass_manager = PassManager::new(&self.context);
        pass_manager.enable_verifier(true);
        pass_manager.add_pass(pass::transform::create_canonicalizer());
        pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
        pass_manager.add_pass(pass::conversion::create_arith_to_llvm());
        pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
        pass_manager.add_pass(pass::conversion::create_func_to_llvm());
        pass_manager.add_pass(pass::conversion::create_index_to_llvm_pass());
        pass_manager.add_pass(pass::conversion::create_mem_ref_to_llvm());
        pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());
        let result = pass_manager.run(module)?;
        Ok(result)
    }
}

pub struct NativeModule<'m> {
    module: Module<'m>,
    registry: ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: MetadataStorage,
}

impl<'m> NativeModule<'m> {
    fn new(
        module: Module<'m>,
        registry: ProgramRegistry<CoreType, CoreLibfunc>,
        metadata: MetadataStorage,
    ) -> Self {
        Self {
            module,
            registry,
            metadata,
        }
    }

    pub fn get_required_init_gas(&self, fn_id: &FunctionId) -> Option<u64> {
        if let Some(gas_metadata) = self.metadata.get::<GasMetadata>() {
            gas_metadata.get_initial_required_gas(&fn_id)
        } else {
            None
        }
    }

    pub fn insert_metadata<T>(&mut self, meta: T) -> Option<&mut T>
    where
        T: Any,
    {
        self.metadata.insert(meta)
    }

    pub fn get_metadata<T>(&self) -> Option<&T>
    where
        T: Any,
    {
        self.metadata.get::<T>()
    }
}

pub struct NativeExecutor<'m> {
    engine: ExecutionEngine,
    native_module: NativeModule<'m>,
}

impl<'m> NativeExecutor<'m> {
    pub fn new(native_module: NativeModule<'m>) -> Self {
        let engine = create_engine(&native_module.module);
        Self {
            engine,
            native_module,
        }
    }

    pub fn get_module(&self) -> &NativeModule<'m> {
        &self.native_module
    }

    pub fn get_program_registry(&self) -> &ProgramRegistry<CoreType, CoreLibfunc> {
        &self.native_module.registry
    }

    pub fn execute<'de, D, S>(
        &self,
        fn_id: &FunctionId,
        params: D,
        returns: S,
        required_init_gas: Option<u64>,
    ) -> Result<S::Ok, Box<JitRunnerError<'de, CoreType, CoreLibfunc, D, S>>>
    where
        D: Deserializer<'de>,
        S: Serializer,
    {
        Ok(execute(
            &self.engine,
            &self.native_module.registry,
            fn_id,
            params,
            returns,
            required_init_gas,
        )?)
    }
}

/// Initialize an MLIR context.
pub fn initialize_mlir() -> Context {
    let context = Context::new();
    context.append_dialect_registry(&{
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        registry
    });
    context.load_all_available_dialects();
    register_all_passes();
    context
}
