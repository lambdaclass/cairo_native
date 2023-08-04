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
    utils,
    values::ValueBuilder,
};
use cairo_lang_compiler::CompilerConfig;
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        GenericLibfunc, GenericType,
    },
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
use std::{fmt, ops::Neg, path::Path, sync::Arc};

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

/// Shortcut to compile and execute a program.
///
/// For short programs this function may suffice, but as the program grows the other interface is
/// preferred since there is some stuff that should be cached, such as the MLIR context and the
/// execution engines for programs that will be run multiple times.
pub fn compile_and_execute<'de, D, S>(
    program: &Path,
    entry_point: &str,
    params: D,
    returns: S,
) -> Result<(), Box<Error<'de, CoreType, CoreLibfunc, D, S>>>
where
    D: Deserializer<'de>,
    S: Serializer,
{
    // Compile the cairo program to sierra.
    let program = &if program
        .extension()
        .map(|x| {
            x.to_ascii_lowercase()
                .to_string_lossy()
                .eq_ignore_ascii_case("cairo")
        })
        .unwrap_or(false)
    {
        cairo_lang_compiler::compile_cairo_project_at_path(
            program,
            CompilerConfig {
                replace_ids: true,
                ..Default::default()
            },
        )
        .unwrap()
    } else {
        let source = std::fs::read_to_string(program).unwrap();
        Arc::new(
            cairo_lang_sierra::ProgramParser::new()
                .parse(&source)
                .unwrap(),
        )
    };

    let function_id = &program
        .funcs
        .iter()
        .find(|x| x.id.debug_name.as_deref() == Some(entry_point))
        .unwrap()
        .id;

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
    let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(program)
        .map_err(|e| Error::Compile(e.into()))?;

    // Make the runtime library available.
    metadata.insert(RuntimeBindingsMeta::default()).unwrap();

    // Gas
    let required_initial_gas = if program
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
    };

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
    utils::register_runtime_symbols(&engine);

    // Execute
    crate::execute::<CoreType, CoreLibfunc, D, S>(
        &engine,
        &registry,
        function_id,
        params,
        returns,
        required_initial_gas,
    )
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

// Parse numeric string into felt, wrapping negatives around the prime modulo.
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

/// Parse any time that can be a bigint to a felt that can be used in the cairo-native input.
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

// Parse a short felt string into felt, wrapping negatives around the prime modulo.
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
