//! The easy to use API to compile and execute cairo programs using the MLIR JIT.
//!
//! Check out the main [`crate`] docs for more information.

use crate::{
    libfuncs::LibfuncBuilder,
    metadata::{
        gas::{GasMetadata, MetadataComputationConfig},
        runtime_bindings::RuntimeBindingsMeta,
        MetadataStorage,
    },
    types::{felt252::PRIME, TypeBuilder},
    utils::register_runtime_symbols,
    values::ValueBuilder,
};
use cairo_lang_sierra::ids::FunctionId;
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        GenericLibfunc, GenericType,
    },
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
use num_bigint::{BigInt, BigUint, Sign};
use serde::{Deserializer, Serializer};
use std::{fmt, ops::Neg};

/// The possible errors encountered when calling [`compile_and_execute`]
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

/// Returns an Option indicating whether a function entrypoint requires an initial gas value or not.
/// Also inserts Gas metadata if needed.
pub fn required_initial_gas(
    program: &Program,
    function_id: &FunctionId,
    metadata: &mut MetadataStorage,
) -> Option<u64> {
    if program
        .type_declarations
        .iter()
        .any(|decl| decl.long_id.generic_id.0.as_str() == "GasBuiltin")
    {
        let gas_metadata = GasMetadata::new(program, MetadataComputationConfig::default());
        let required_initial_gas = { gas_metadata.get_initial_required_gas(function_id) };
        metadata.insert(gas_metadata).unwrap();
        required_initial_gas
    } else {
        None
    }
}

/// Performs all steps to compile a Sierra program into an MLIR context which can later be lowered to LLVM.
// TODO: Rethink this error type to make it simpler.
#[allow(clippy::type_complexity)]
pub fn compile_sierra_to_mlir<'c, 'd, D, S>(
    context: &'c Context,
    program: &Program,
    function_id: &FunctionId,
) -> Result<
    (
        Module<'c>,
        ProgramRegistry<CoreType, CoreLibfunc>,
        Option<u64>,
    ),
    Box<Error<'d, CoreType, CoreLibfunc, D, S>>,
>
where
    D: Deserializer<'d>,
    S: Serializer,
{
    // Create the empty module
    let module = Module::new(Location::unknown(context));

    // Create the metadata storage
    let mut metadata = MetadataStorage::new();

    // Create the Sierra program registry
    let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(program)
        .map_err(|e| Error::Compile(e.into()))?;

    // Make the runtime library available by inserting it into the metadata so it can be later retrieved
    metadata.insert(RuntimeBindingsMeta::default()).unwrap();

    // Check whether the entry point of the program requires an initial gas value
    let required_initial_gas = required_initial_gas(program, function_id, &mut metadata);

    crate::compile(context, &module, program, &registry, &mut metadata, None)
        .map_err(Error::Compile)?;

    Ok((module, registry, required_initial_gas))
}

/// Given an MLIR context and module, lowers the operations to LLVM IR.
// TODO: We should check what error is it best to return here.
pub fn lower_mlir_to_llvm(context: &Context, module: &mut Module) -> Result<(), melior::Error> {
    let pass_manager = PassManager::new(context);
    pass_manager.enable_verifier(true);
    pass_manager.add_pass(pass::transform::create_canonicalizer());
    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
    pass_manager.add_pass(pass::conversion::create_arith_to_llvm());
    pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
    pass_manager.add_pass(pass::conversion::create_func_to_llvm());
    pass_manager.add_pass(pass::conversion::create_index_to_llvm_pass());
    pass_manager.add_pass(pass::conversion::create_mem_ref_to_llvm());
    pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());
    pass_manager.run(module)
}

/// Parse a numeric string into felt, wrapping negatives around the prime modulo.
pub fn felt252_str(value: &str) -> [u32; 8] {
    let value = value
        .parse::<BigInt>()
        .expect("value must be a digit number");
    let value = match value.sign() {
        Sign::Minus => &*PRIME - value.neg().to_biguint().unwrap(),
        _ => value.to_biguint().unwrap(),
    };

    let mut u32_digits = value.to_u32_digits();
    u32_digits.resize(8, 0);
    u32_digits.try_into().unwrap()
}

/// Parse any type that can be a bigint to a felt that can be used in the cairo-native input.
pub fn felt252_bigint(value: impl Into<BigInt>) -> [u32; 8] {
    let value: BigInt = value.into();
    let value = match value.sign() {
        Sign::Minus => &*PRIME - value.neg().to_biguint().unwrap(),
        _ => value.to_biguint().unwrap(),
    };

    let mut u32_digits = value.to_u32_digits();
    u32_digits.resize(8, 0);
    u32_digits.try_into().unwrap()
}

/// Parse a short string into a felt that can be used in the cairo-native input.
pub fn felt252_short_str(value: &str) -> [u32; 8] {
    let values: Vec<_> = value
        .chars()
        .filter(|&c| c.is_ascii())
        .map(|c| c as u8)
        .collect();

    let mut digits = BigUint::from_bytes_be(&values).to_u32_digits();
    digits.resize(8, 0);
    digits.try_into().unwrap()
}

/// Creates all the structures needed to compile and create the JIT engine.
#[allow(clippy::type_complexity)]
pub fn create_compiler(
    program: &Program,
) -> Result<
    (
        Context,
        Module,
        ProgramRegistry<CoreType, CoreLibfunc>,
        MetadataStorage,
    ),
    Box<dyn std::error::Error>,
> {
    // Initialize MLIR.
    let context = initialize_mlir();

    // Compile the program.
    let module = Module::new(Location::unknown(&context));
    let mut metadata = MetadataStorage::new();
    let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(program)?;

    // Make the runtime library available.
    metadata
        .insert(RuntimeBindingsMeta::default())
        .ok_or("Could not insert runtime library")?;

    Ok((context, module, registry, metadata))
}

/// Creates the execution engine, with all symbols registered.
pub fn create_engine(module: &Module) -> ExecutionEngine {
    // Create the JIT engine.
    let engine = ExecutionEngine::new(module, 3, &[], false);

    #[cfg(feature = "with-runtime")]
    register_runtime_symbols(&engine);

    engine
}
