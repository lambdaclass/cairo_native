#![feature(arc_unwrap_or_clone)]
#![feature(nonzero_ops)]
#![feature(strict_provenance)]

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
    metadata::MetadataStorage,
    types::TypeBuilder,
    utils::{debug_with, generate_function_name},
    values::ValueBuilder,
    DebugInfo,
};
use std::{
    alloc::Layout,
    convert::Infallible,
    ffi::OsStr,
    fs,
    iter::once,
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

    let mut params_io = Vec::new();
    for param in &entry_point.signature.param_types {
        let concrete_type = registry.get_type(param)?;

        // Deserialize every argument into a value.
        params_io.push(match concrete_type {
            CoreTypeConcrete::Array(_) => todo!(),
            CoreTypeConcrete::Bitwise(_) => todo!(),
            CoreTypeConcrete::Box(_) => todo!(),
            CoreTypeConcrete::EcOp(_) => todo!(),
            CoreTypeConcrete::EcPoint(_) => todo!(),
            CoreTypeConcrete::EcState(_) => todo!(),
            CoreTypeConcrete::Felt252(_) => todo!(),
            CoreTypeConcrete::GasBuiltin(_) => todo!(),
            CoreTypeConcrete::BuiltinCosts(_) => todo!(),
            CoreTypeConcrete::Uint8(_) => todo!(),
            CoreTypeConcrete::Uint16(_) => todo!(),
            CoreTypeConcrete::Uint32(_) => todo!(),
            CoreTypeConcrete::Uint64(_) => todo!(),
            CoreTypeConcrete::Uint128(_) => todo!(),
            CoreTypeConcrete::Uint128MulGuarantee(_) => todo!(),
            CoreTypeConcrete::NonZero(_) => todo!(),
            CoreTypeConcrete::Nullable(_) => todo!(),
            CoreTypeConcrete::RangeCheck(_) => todo!(),
            CoreTypeConcrete::Uninitialized(_) => todo!(),
            CoreTypeConcrete::Enum(_) => todo!(),
            CoreTypeConcrete::Struct(_) => todo!(),
            CoreTypeConcrete::Felt252Dict(_) => todo!(),
            CoreTypeConcrete::Felt252DictEntry(_) => todo!(),
            CoreTypeConcrete::SquashedFelt252Dict(_) => todo!(),
            CoreTypeConcrete::Pedersen(_) => todo!(),
            CoreTypeConcrete::Poseidon(_) => todo!(),
            CoreTypeConcrete::Span(_) => todo!(),
            CoreTypeConcrete::StarkNet(_) => todo!(),
            CoreTypeConcrete::SegmentArena(_) => todo!(),
            CoreTypeConcrete::Snapshot(_) => todo!(),
        });
    }

    let mut rets_io = Vec::<usize>::new();
    let mut rets_layout: Option<Layout> = None;
    let mut rets_layout_is_complex = entry_point.signature.ret_types.len() > 1;
    for ret in &entry_point.signature.ret_types {
        let concrete_type = registry.get_type(ret)?;

        // Generate the layout of a struct with every return value.
        let layout = concrete_type.layout(&registry);
        let (layout, offset) = match rets_layout {
            Some(acc) => acc.extend(layout)?,
            None => (layout, 0),
        };

        rets_io.push(offset);
        rets_layout = Some(layout);
        rets_layout_is_complex |= match concrete_type {
            CoreTypeConcrete::Array(_) => todo!(),
            CoreTypeConcrete::Bitwise(_) => todo!(),
            CoreTypeConcrete::Box(_) => todo!(),
            CoreTypeConcrete::EcOp(_) => todo!(),
            CoreTypeConcrete::EcPoint(_) => todo!(),
            CoreTypeConcrete::EcState(_) => todo!(),
            CoreTypeConcrete::Felt252(_) => false,
            CoreTypeConcrete::GasBuiltin(_) => todo!(),
            CoreTypeConcrete::BuiltinCosts(_) => todo!(),
            CoreTypeConcrete::Uint8(_) => todo!(),
            CoreTypeConcrete::Uint16(_) => todo!(),
            CoreTypeConcrete::Uint32(_) => todo!(),
            CoreTypeConcrete::Uint64(_) => todo!(),
            CoreTypeConcrete::Uint128(_) => todo!(),
            CoreTypeConcrete::Uint128MulGuarantee(_) => todo!(),
            CoreTypeConcrete::NonZero(_) => todo!(),
            CoreTypeConcrete::Nullable(_) => todo!(),
            CoreTypeConcrete::RangeCheck(_) => todo!(),
            CoreTypeConcrete::Uninitialized(_) => todo!(),
            CoreTypeConcrete::Enum(_) => todo!(),
            CoreTypeConcrete::Struct(_) => true,
            CoreTypeConcrete::Felt252Dict(_) => todo!(),
            CoreTypeConcrete::Felt252DictEntry(_) => todo!(),
            CoreTypeConcrete::SquashedFelt252Dict(_) => todo!(),
            CoreTypeConcrete::Pedersen(_) => todo!(),
            CoreTypeConcrete::Poseidon(_) => todo!(),
            CoreTypeConcrete::Span(_) => todo!(),
            CoreTypeConcrete::StarkNet(_) => todo!(),
            CoreTypeConcrete::SegmentArena(_) => todo!(),
            CoreTypeConcrete::Snapshot(_) => todo!(),
        }
    }

    let rets_ptr = arena
        .alloc_layout(rets_layout.unwrap_or(Layout::from_size_align(0, 1)?))
        .cast::<()>();
    let mut invoke_io = once(if rets_layout_is_complex {
        arena.alloc(rets_ptr.as_ptr()) as *mut *mut () as *mut ()
    } else {
        rets_ptr.as_ptr()
    })
    .chain(params_io)
    .collect::<Vec<_>>();

    assert_ne!(rets_ptr.as_ptr(), invoke_io[0]);

    // Invoke the entry point.
    unsafe {
        engine.invoke_packed(&generate_function_name(&entry_point.id), &mut invoke_io)?;
    }

    // Print returned values.
    let mut layout: Option<Layout> = None;
    for ty in &entry_point.signature.ret_types {
        let concrete_type = registry.get_type(ty)?;

        let ty_layout = concrete_type.layout(&registry);
        let (new_layout, offset) = match layout {
            Some(layout) => layout.extend(ty_layout)?,
            None => (ty_layout, 0),
        };
        layout = Some(new_layout);

        let value_ptr = rets_ptr.map_addr(|addr| unsafe { addr.unchecked_add(offset) });
        match args.outputs {
            StdioOrPath::Stdio => println!(
                "{:#?}",
                debug_with(|f| unsafe { concrete_type.debug_fmt(f, ty, &registry, value_ptr) })
            ),
            StdioOrPath::Path(_) => todo!(),
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

    #[clap(short = 'i', long = "inputs", value_parser = parse_io)]
    inputs: Option<StdioOrPath>,
    #[clap(short = 'o', long = "outputs", value_parser = parse_io, default_value = "-")]
    outputs: StdioOrPath,

    #[clap(short = 'g', long = "available-gas")]
    available_gas: Option<usize>,
}

#[derive(Clone, Debug)]
enum StdioOrPath {
    Stdio,
    Path(PathBuf),
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

fn parse_io(input: &str) -> Result<StdioOrPath, String> {
    Ok(if input == "-" {
        StdioOrPath::Stdio
    } else {
        StdioOrPath::Path(match Path::new(input).extension().and_then(OsStr::to_str) {
            Some("json") => input.into(),
            _ => return Err("Input path expected to have `json` as its extension.".to_string()),
        })
    })
}
