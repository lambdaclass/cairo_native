#![feature(arc_unwrap_or_clone)]

use bumpalo::Bump;
use cairo_lang_compiler::{
    compile_prepared_db, db::RootDatabase, diagnostics::DiagnosticsReporter,
    project::setup_project, CompilerConfig,
};
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType, CoreTypeConcrete},
    ids::FunctionId,
    program::Program,
    program_registry::ProgramRegistry,
    ProgramParser,
};
use clap::Parser;
use melior::{
    dialect::DialectRegistry,
    ir::{Location, Module},
    pass::{self, PassManager},
    utility::{register_all_dialects, register_all_passes},
    Context, ExecutionEngine,
};
use sierra2mlir::{
    generate_function_name,
    metadata::MetadataStorage,
    values::{DebugWrapper, ValueBuilder},
    DebugInfo,
};
use std::{
    cell::RefCell,
    convert::Infallible,
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};
use tracing_subscriber::{EnvFilter, FmtSubscriber};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command-line arguments.
    let args = CmdLine::parse();

    // Configure logging and error handling.
    tracing::subscriber::set_global_default(
        FmtSubscriber::builder()
            .with_env_filter(EnvFilter::from_default_env())
            .finish(),
    )?;

    // Load the program.
    let (program, _debug_info) = load_program(Path::new(&args.input))?;

    let entry_point = match program
        .funcs
        .iter()
        .find(|x| x.id.debug_name == args.entry_point.debug_name || x.id == args.entry_point)
    {
        Some(x) => x,
        None => {
            // TODO: Use a proper error.
            eprintln!("Entry point `{}` not found in program.", args.entry_point);
            return Ok(());
        }
    };

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
    let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program)?;

    sierra2mlir::compile::<CoreType, CoreLibfunc>(
        &context,
        &module,
        &program,
        &registry,
        &mut metadata,
    )?;

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

    pass_manager.run(&mut module)?;

    // Create the JIT engine.
    let engine = ExecutionEngine::new(&module, 3, &[], false);

    // Initialize arguments and return values.
    let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program)?;

    let arena = Bump::new();
    let mut invoke_io = Vec::new();
    for param in &entry_point.signature.param_types {
        let concrete_type = registry.get_type(param)?;

        match concrete_type {
            // Virtual types (we don't use them, they exist for the VM).
            CoreTypeConcrete::Bitwise(_)
            | CoreTypeConcrete::BuiltinCosts(_)
            | CoreTypeConcrete::RangeCheck(_)
            | CoreTypeConcrete::Uninitialized(_)
            | CoreTypeConcrete::Pedersen(_)
            | CoreTypeConcrete::Poseidon(_) => {
                invoke_io.push(concrete_type.alloc(
                    &arena,
                    &context,
                    &module,
                    &registry,
                    &mut metadata,
                ));
            }

            // Types that require special handling.
            CoreTypeConcrete::GasBuiltin(_) => {
                let available_gas = args
                    .available_gas
                    .expect("Gas is required, but no limit has been provided.");

                invoke_io.push(concrete_type.parsed(
                    &arena,
                    &context,
                    &module,
                    &registry,
                    &mut metadata,
                    &available_gas.to_string(),
                )?);
            }

            // Unhandled types.
            CoreTypeConcrete::Box(_)
            | CoreTypeConcrete::EcOp(_)
            | CoreTypeConcrete::EcPoint(_)
            | CoreTypeConcrete::EcState(_)
            | CoreTypeConcrete::Uint128MulGuarantee(_)
            | CoreTypeConcrete::Felt252Dict(_)
            | CoreTypeConcrete::Felt252DictEntry(_)
            | CoreTypeConcrete::SquashedFelt252Dict(_)
            | CoreTypeConcrete::Span(_)
            | CoreTypeConcrete::StarkNet(_)
            | CoreTypeConcrete::SegmentArena(_)
            | CoreTypeConcrete::Snapshot(_) => todo!("unhandled type"),

            // Actual input types.
            CoreTypeConcrete::Array(_)
            | CoreTypeConcrete::Felt252(_)
            | CoreTypeConcrete::Uint8(_)
            | CoreTypeConcrete::Uint16(_)
            | CoreTypeConcrete::Uint32(_)
            | CoreTypeConcrete::Uint64(_)
            | CoreTypeConcrete::Uint128(_)
            | CoreTypeConcrete::NonZero(_)
            | CoreTypeConcrete::Nullable(_)
            | CoreTypeConcrete::Enum(_)
            | CoreTypeConcrete::Struct(_) => todo!(),
        }
    }
    for ret in &entry_point.signature.ret_types {
        let concrete_type = registry.get_type(ret)?;
        invoke_io.push(concrete_type.alloc(&arena, &context, &module, &registry, &mut metadata));
    }

    unsafe {
        engine.invoke_packed(&generate_function_name(&entry_point.id), &mut invoke_io)?;
    }

    // Print returned values.
    for (ptr, ty) in invoke_io
        .into_iter()
        .skip(entry_point.signature.param_types.len())
        .zip(&entry_point.signature.ret_types)
    {
        let concrete_type = registry.get_type(ty).unwrap();
        let wrapper = DebugWrapper {
            inner: concrete_type,
            context: &context,
            module: &module,
            registry: &registry,
            metadata: RefCell::new(&mut metadata),
            id: ty,
            source: ptr,
        };

        match concrete_type {
            // Virtual types (we don't use them, they exist for the VM).
            CoreTypeConcrete::Bitwise(_)
            | CoreTypeConcrete::BuiltinCosts(_)
            | CoreTypeConcrete::RangeCheck(_)
            | CoreTypeConcrete::Uninitialized(_)
            | CoreTypeConcrete::Pedersen(_)
            | CoreTypeConcrete::Poseidon(_) => {}

            // Types that require special handling.
            CoreTypeConcrete::GasBuiltin(_) => {
                println!("Remaining gas: {wrapper:?}");
            }

            // Unhandled types.
            CoreTypeConcrete::Box(_)
            | CoreTypeConcrete::EcOp(_)
            | CoreTypeConcrete::EcPoint(_)
            | CoreTypeConcrete::EcState(_)
            | CoreTypeConcrete::Uint128MulGuarantee(_)
            | CoreTypeConcrete::Felt252Dict(_)
            | CoreTypeConcrete::Felt252DictEntry(_)
            | CoreTypeConcrete::SquashedFelt252Dict(_)
            | CoreTypeConcrete::Span(_)
            | CoreTypeConcrete::StarkNet(_)
            | CoreTypeConcrete::SegmentArena(_)
            | CoreTypeConcrete::Snapshot(_) => todo!("unhandled type"),

            // Actual input types.
            CoreTypeConcrete::Array(_)
            | CoreTypeConcrete::Felt252(_)
            | CoreTypeConcrete::Uint8(_)
            | CoreTypeConcrete::Uint16(_)
            | CoreTypeConcrete::Uint32(_)
            | CoreTypeConcrete::Uint64(_)
            | CoreTypeConcrete::Uint128(_)
            | CoreTypeConcrete::NonZero(_)
            | CoreTypeConcrete::Nullable(_)
            | CoreTypeConcrete::Enum(_)
            | CoreTypeConcrete::Struct(_) => {
                println!("{wrapper:?}");
            }
        }
    }

    Ok(())
}

fn load_program(path: &Path) -> Result<(Program, Option<DebugInfo>), Box<dyn std::error::Error>> {
    Ok(match path.extension().and_then(OsStr::to_str) {
        Some("cairo") => {
            let mut db = RootDatabase::builder().detect_corelib().build()?;
            let main_crate_ids = setup_project(&mut db, path)?;
            let program = Arc::unwrap_or_clone(compile_prepared_db(
                &mut db,
                main_crate_ids,
                CompilerConfig {
                    replace_ids: true,
                    ..Default::default()
                },
            )?);

            let debug_info = DebugInfo::extract(&db, &program).map_err(|_| {
                let mut buffer = String::new();
                assert!(DiagnosticsReporter::write_to_string(&mut buffer).check(&db));
                buffer
            })?;

            (program, Some(debug_info))
        }
        Some("sierra") => {
            let program_src = fs::read_to_string(path)?;

            let program_parser = ProgramParser::new();
            let program = program_parser
                .parse(&program_src)
                .map_err(|e| e.map_token(|t| t.to_string()))?;

            (program, None)
        }
        _ => unreachable!(),
    })
}

#[derive(Clone, Debug, Parser)]
struct CmdLine {
    #[clap(value_parser = parse_input)]
    input: PathBuf,

    #[clap(value_parser = parse_entry_point)]
    entry_point: FunctionId,

    #[clap(short = 'g', long = "available-gas")]
    available_gas: Option<usize>,
}

fn parse_input(input: &str) -> Result<PathBuf, String> {
    Ok(match Path::new(input).extension().and_then(OsStr::to_str) {
        Some("cairo" | "sierra") => input.into(),
        _ => {
            return Err(
                "Input path expected to have either `cairo` or `sierra` as its extension."
                    .to_string(),
            )
        }
    })
}

fn parse_entry_point(input: &str) -> Result<FunctionId, Infallible> {
    Ok(match input.parse::<u64>() {
        Ok(id) => FunctionId::new(id),
        Err(_) => FunctionId::from_string(input),
    })
}
