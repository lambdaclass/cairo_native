use crate::{
    libfuncs::LibfuncBuilder,
    metadata::{runtime_bindings::RuntimeBindingsMeta, MetadataStorage},
    types::TypeBuilder,
    utils::register_runtime_symbols,
    values::ValueBuilder,
};
use cairo_lang_sierra::{
    extensions::{GenericLibfunc, GenericType},
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
use serde::{Deserializer, Serializer};
use std::fmt;

pub enum Error<'de, TType, TLibfunc, D, S>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc> + ValueBuilder<TType, TLibfunc>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc>,
    D: Deserializer<'de>,
    S: Serializer,
{
    Compile(crate::error::CompileError<TType, TLibfunc>),
    JitRunner(crate::error::JitRunnerError<'de, TType, TLibfunc, D, S>),
}

impl<'de, TType, TLibfunc, D, S> fmt::Debug for Error<'de, TType, TLibfunc, D, S>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc> + ValueBuilder<TType, TLibfunc>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc>,
    D: Deserializer<'de>,
    S: Serializer,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compile(x) => fmt::Debug::fmt(x, f),
            Self::JitRunner(x) => fmt::Debug::fmt(x, f),
        }
    }
}

impl<'de, TType, TLibfunc, D, S> fmt::Display for Error<'de, TType, TLibfunc, D, S>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc> + ValueBuilder<TType, TLibfunc>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc>,
    D: Deserializer<'de>,
    S: Serializer,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compile(x) => fmt::Display::fmt(x, f),
            Self::JitRunner(x) => fmt::Display::fmt(x, f),
        }
    }
}

impl<'de, TType, TLibfunc, D, S> From<crate::error::CompileError<TType, TLibfunc>>
    for Error<'de, TType, TLibfunc, D, S>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc> + ValueBuilder<TType, TLibfunc>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc>,
    D: Deserializer<'de>,
    S: Serializer,
{
    fn from(value: crate::error::CompileError<TType, TLibfunc>) -> Self {
        Self::Compile(value)
    }
}

impl<'de, TType, TLibfunc, D, S> From<crate::error::JitRunnerError<'de, TType, TLibfunc, D, S>>
    for Error<'de, TType, TLibfunc, D, S>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc> + ValueBuilder<TType, TLibfunc>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc>,
    D: Deserializer<'de>,
    S: Serializer,
{
    fn from(value: crate::error::JitRunnerError<'de, TType, TLibfunc, D, S>) -> Self {
        Self::JitRunner(value)
    }
}

/// Shortcut to compile and execute a program.
///
/// For short programs this function may suffice, but as the program grows the other interface is
/// preferred since there is some stuff that should be cached, such as the MLIR context and the
/// execution engines for programs that will be run multiple times.
pub fn compile_and_execute<'de, TType, TLibfunc, D, S>(
    program: &Program,
    function_id: &FunctionId,
    params: D,
    returns: S,
) -> Result<(), Error<'de, TType, TLibfunc, D, S>>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc> + ValueBuilder<TType, TLibfunc>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc>,
    D: Deserializer<'de>,
    S: Serializer,
{
    // Initialize MLIR.
    let context = Context::new();
    context.append_dialect_registry(&{
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        registry
    });
    context.load_all_available_dialects();

    register_all_passes();

    // Compile the program.
    let mut module = Module::new(Location::unknown(&context));
    let mut metadata = MetadataStorage::new();
    let registry =
        ProgramRegistry::<TType, TLibfunc>::new(program).map_err(|e| Error::Compile(e.into()))?;

    // Make the runtime library available.
    metadata.insert(RuntimeBindingsMeta::default()).unwrap();

    crate::compile(&context, &module, program, &registry, &mut metadata, None)
        .map_err(Error::Compile)?;

    // Lower to LLVM.
    let pass_manager = PassManager::new(&context);
    pass_manager.enable_verifier(true);
    pass_manager.add_pass(pass::transform::create_canonicalizer());

    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());

    pass_manager.add_pass(pass::conversion::create_arith_to_llvm());
    pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
    pass_manager.add_pass(pass::conversion::create_func_to_llvm());
    pass_manager.add_pass(pass::conversion::create_index_to_llvm_pass());
    pass_manager.add_pass(pass::conversion::create_mem_ref_to_llvm());
    pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());

    pass_manager
        .run(&mut module)
        .map_err(|e| Error::JitRunner(e.into()))?;

    // Create the JIT engine.
    let engine = ExecutionEngine::new(&module, 3, &[], false);

    #[cfg(feature = "with-runtime")]
    register_runtime_symbols(&engine);

    // Execute
    crate::execute::<TType, TLibfunc, D, S>(&engine, &registry, function_id, params, returns)
        .unwrap_or_else(|e| match &*e {
            crate::error::jit_engine::ErrorImpl::DeserializeError(_) => {
                panic!(
                    "Expected inputs with signature: ({})",
                    registry
                        .get_function(function_id)
                        .unwrap()
                        .signature
                        .param_types
                        .iter()
                        .map(ToString::to_string)
                        .intersperse_with(|| ", ".to_string())
                        .collect::<String>()
                )
            }
            e => panic!("{:?}", e),
        });

    Ok(())
}
