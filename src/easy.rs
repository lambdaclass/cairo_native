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
    utils::{self, register_runtime_symbols},
    values::ValueBuilder,
};
use cairo_lang_compiler::CompilerConfig;
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        GenericLibfunc, GenericType,
    },
    program::{GenFunction, Program, StatementIdx},
    program_registry::ProgramRegistry,
};
use cairo_lang_sierra::ids::FunctionId;
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

/// Compile a cairo program found at the given path to sierra.
pub fn cairo_to_sierra(program: &Path) -> Arc<Program> {
    if program
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
    }
}

/// Given a string representing a function name, searches in the program for the id corresponding to said function, and returns a reference to it.
pub fn find_function_id<'a >(program: &'a Program, function_name: &str) -> &'a FunctionId {
    &program
        .funcs
        .iter()
        .find(|x| x.id.debug_name.as_deref() == Some(function_name))
        .unwrap()
        .id
}

pub fn initialize_mlir() -> Context {
    let context = Context::new();
    context.append_dialect_registry(&{
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        registry
    });
    context.load_all_available_dialects();

    register_all_passes();

    return context
}

/// Returns an Option indicating whether a function entrypoint requires an initial gas value.
pub fn required_initial_gas<'p, 'm>(program: &'p Program, function_id: &'p FunctionId, metadata: &'m mut MetadataStorage) -> Option<u64> {
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

pub fn compile_sierra_to_mlir<'c, 'p, 'd, D, S>(context: &'c Context, program: &'p Program, function_id: &'p FunctionId) 
-> Result<( Module<'c>, ProgramRegistry<CoreType, CoreLibfunc>, Option<u64> ), Box<Error<'d, CoreType, CoreLibfunc, D, S>>> 
where
    D: Deserializer<'d>,
    S: Serializer,
{

    // Create the empty module
    let module = Module::new(Location::unknown(&context));

    // Create the metadata storage
    let mut metadata = MetadataStorage::new();

    // Create the Sierra program registry
    let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(program)
        .map_err(|e| Error::Compile(e.into()))?;

    // Make the runtime library available by inserting it into the metadata so it can be later retrieved
    metadata.insert(RuntimeBindingsMeta::default()).unwrap();

    // Check whether the entry point of the program requires an initial gas value
    let required_initial_gas = required_initial_gas(program, function_id, &mut metadata);

    crate::compile(&context, &module, program, &registry, &mut metadata, None)
        .map_err(Error::Compile)?;

    return Ok( (module, registry, required_initial_gas) )
}

pub fn lower_mlir_to_llvm<'c, 'd, D, S>(context: &'c Context, module: &'c mut Module) 
    -> Result<(), Box<Error<'d, CoreType, CoreLibfunc, D, S>>> 
where
    D: Deserializer<'d>,
    S: Serializer,
{
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
    Ok(())
}

/// Shortcut to compile and execute a program.
///
/// For short programs this function may suffice, but as the program grows the other interface is
/// preferred since there is some stuff that should be cached, such as the MLIR context and the
/// execution engines for programs that will be run multiple times.
pub fn compile_and_execute<'de, D, S>(
    program_path: &Path,
    entry_point: &str,
    params: D,
    returns: S,
) -> Result<(), Box<Error<'de, CoreType, CoreLibfunc, D, S>>>
where
    D: Deserializer<'de>,
    S: Serializer,
{
    // Compile the cairo program to sierra.
    let program = cairo_to_sierra(program_path);
    let function_id = find_function_id(&program, entry_point);
    
    // Initialize MLIR.
    let context = initialize_mlir();

    // Compile sierra to MLIR
    let (mut module, registry, required_initial_gas) = compile_sierra_to_mlir(&context, &program, function_id)?;

    // Lower MLIR to LLVM
    lower_mlir_to_llvm(&context, &mut module);

    // Create the JIT engine.
    let engine = ExecutionEngine::new(&module, 3, &[], false);

    #[cfg(feature = "with-runtime")]
    utils::register_runtime_symbols(&engine);

    // Execute the program
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

/// Returns the given entry point.
pub fn find_entry_point<'a>(
    program: &'a Program,
    entry_point: &str,
) -> Option<&'a GenFunction<StatementIdx>> {
    program
        .funcs
        .iter()
        .find(|x| x.id.debug_name.as_deref() == Some(entry_point))
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
    let context = Context::new();
    context.append_dialect_registry(&{
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        registry
    });
    context.load_all_available_dialects();

    register_all_passes();

    // Compile the program.
    let module = Module::new(Location::unknown(&context));
    let mut metadata = MetadataStorage::new();
    let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(program)?;

    // Make the runtime library available.
    metadata.insert(RuntimeBindingsMeta::default()).unwrap();

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

/// Returns the required initial gas, also inserts de Gas metadata if needed.
pub fn get_required_initial_gas(
    program: &Program,
    metadata: &mut MetadataStorage,
    entry_point: &GenFunction<StatementIdx>,
) -> Option<u64> {
    if program
        .type_declarations
        .iter()
        .any(|decl| decl.long_id.generic_id.0.as_str() == "GasBuiltin")
    {
        let gas_metadata = GasMetadata::new(program, MetadataComputationConfig::default());

        let required_initial_gas = { gas_metadata.get_initial_required_gas(&entry_point.id) };
        metadata.insert(gas_metadata).unwrap();
        required_initial_gas
    } else {
        None
    }
}

/// Runs the needed MLIR passes on the module to execute with the JIT.
pub fn run_passes(
    context: &Context,
    module: &mut Module,
) -> std::result::Result<(), melior::Error> {
    // Lower to LLVM.
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
