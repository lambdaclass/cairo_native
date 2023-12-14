use cairo_lang_compiler::{
    compile_prepared_db, db::RootDatabase, project::setup_project, CompilerConfig,
};
use cairo_lang_sierra::{ids::FunctionId, program::Program, ProgramParser};
use cairo_native::{context::NativeContext, executor::NativeExecutor, values::JITValue};
use clap::Parser;
use itertools::Itertools;
use starknet_types_core::felt::Felt;
use std::{
    borrow::Cow,
    convert::Infallible,
    ffi::OsStr,
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
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
    let sierra_program = load_program(Path::new(&args.input))?;

    let entry_point = match sierra_program
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

    let native_context = NativeContext::new();

    // Compile the sierra program into a MLIR module.
    let native_program = native_context.compile(&sierra_program).unwrap();
    let native_executor = NativeExecutor::new(native_program);

    // Initialize arguments and return values.
    let params_input = match args.inputs {
        Some(StdioOrPath::Stdio) => Cow::Owned(io::read_to_string(io::stdin())?),
        Some(StdioOrPath::Path(path)) => Cow::Owned(fs::read_to_string(path)?),
        None => Cow::Borrowed(""),
    };

    let params = params_input
        .split_whitespace()
        .map(|x| {
            JITValue::Felt252(Felt::from_dec_str(x).expect("input parameter is not a valid Felt"))
        })
        .collect_vec();

    let result = native_executor
        .execute(&entry_point.id, &params, Some(u64::MAX.into()))
        .unwrap();

    match args.outputs {
        Some(StdioOrPath::Stdio) => {
            println!("{:#?}", result);
        }
        Some(StdioOrPath::Path(path)) => {
            let mut file = File::create(path)?;
            writeln!(file, "{:#?}", result)?;
        }
        None => {
            if args.print_outputs {
                todo!()
            }
        }
    }

    Ok(())
}

fn load_program(path: &Path) -> Result<Program, Box<dyn std::error::Error>> {
    Ok(match path.extension().and_then(OsStr::to_str) {
        Some("cairo") => {
            let mut db = RootDatabase::builder().detect_corelib().build()?;
            let main_crate_ids = setup_project(&mut db, path)?;
            (*compile_prepared_db(
                &mut db,
                main_crate_ids,
                CompilerConfig {
                    replace_ids: true,
                    ..Default::default()
                },
            )?)
            .clone()
        }
        Some("sierra") => {
            let program_src = fs::read_to_string(path)?;

            let program_parser = ProgramParser::new();
            program_parser
                .parse(&program_src)
                .map_err(|e| e.map_token(|t| t.to_string()))?
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
    #[clap(short = 'o', long = "outputs", value_parser = parse_io)]
    outputs: Option<StdioOrPath>,
    #[clap(short = 'p', long = "print-outputs")]
    print_outputs: bool,
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
