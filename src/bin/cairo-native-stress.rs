use std::fmt::Display;
use std::fs::create_dir_all;
use std::hash::Hash;
use std::path::Path;
use std::{collections::HashMap, fs, rc::Rc, time::Instant};

use cairo_lang_sierra::ids::FunctionId;
use cairo_lang_sierra::program::Program;
use cairo_lang_sierra::program_registry::ProgramRegistry;
use cairo_lang_starknet::compile::compile_path;
use cairo_native::metadata::gas::GasMetadata;
use cairo_native::utils::SHARED_LIBRARY_EXT;
use cairo_native::{
    context::NativeContext, executor::AotNativeExecutor, starknet::DummySyscallHandler,
    utils::find_entry_point_by_idx,
};
use cairo_native::{module_to_object, object_to_shared_lib, OptLevel};
use clap::Parser;
use libloading::Library;
use tracing::{info, info_span, trace};
use tracing_subscriber::{EnvFilter, FmtSubscriber};

#[derive(Parser, Debug)]
struct CliArgs {
    /// Amount of iterations to perform
    iterations: u32,
}

fn main() {
    let cli_args = CliArgs::parse();

    tracing::subscriber::set_global_default(
        FmtSubscriber::builder()
            .with_env_filter(EnvFilter::from_default_env())
            .finish(),
    )
    .expect("failed to set global tracing subscriber");

    let before_stress_test = Instant::now();

    let native_context = NativeContext::new();
    let mut cache = AotProgramCache::new(&native_context);

    let (entry_point, program) = {
        let before_generate = Instant::now();
        let initial_program = generate_initial_program();
        let elapsed = before_generate.elapsed().as_millis();
        trace!("generated test program, took {elapsed}ms");
        initial_program
    };

    for round in 0..cli_args.iterations {
        let _enter_span = info_span!("round");

        if cache.get(&round).is_some() {
            panic!("all keys should be different")
        }

        let executor = {
            let before_compile = Instant::now();
            let executor = cache.compile_and_insert(round, &program, cairo_native::OptLevel::None);
            let elapsed = before_compile.elapsed().as_millis();
            trace!("compiled test program, took {elapsed}ms");
            executor
        };

        let execution_result = {
            let now = Instant::now();
            let execution_result = executor
                .invoke_contract_dynamic(&entry_point, &[], Some(u128::MAX), DummySyscallHandler)
                .expect("failed to execute contract");
            let elapsed = now.elapsed().as_millis();
            trace!("executed test program, took {elapsed}ms");
            execution_result
        };

        assert!(
            execution_result.failure_flag == false,
            "contract execution had failure flag set"
        );

        info!(
            "Finished round {round} with result {}",
            execution_result.return_values[0]
        );
    }

    let elapsed = before_stress_test.elapsed().as_millis();
    info!("finished stress test, took {elapsed}ms");
}

fn generate_initial_program() -> (FunctionId, cairo_lang_sierra::program::Program) {
    let program_str = format!(
        "\
#[starknet::contract]
mod Contract {{
    #[storage]
    struct Storage {{}}

    #[external(v0)]
    fn main(self: @ContractState) -> felt252 {{
        return 252;
    }}
}}
"
    );

    let mut program_file = tempfile::Builder::new()
        .prefix("test_")
        .suffix(".cairo")
        .tempfile()
        .expect("failed to create temporary file for cairo test program");
    fs::write(&mut program_file, program_str).expect("failed to write cairo test file");

    let contract_class = compile_path(program_file.path(), None, Default::default())
        .expect("failed to compile cairo contract");

    let program = contract_class
        .extract_sierra_program()
        .expect("failed to extract sierra program");

    let entry_point_idx = contract_class
        .entry_points_by_type
        .external
        .first()
        .expect("contrat should have at least one entrypoint")
        .function_idx;

    let entry_point = find_entry_point_by_idx(&program, entry_point_idx)
        .expect("failed to find entrypoint")
        .id
        .clone();

    (entry_point, program)
}

struct AotProgramCache<'a, K>
where
    K: PartialEq + Eq + Hash + Display,
{
    context: &'a NativeContext,
    cache: HashMap<K, Rc<AotNativeExecutor>>,
}

impl<'a, K> AotProgramCache<'a, K>
where
    K: PartialEq + Eq + Hash + Display,
{
    pub fn new(context: &'a NativeContext) -> Self {
        Self {
            context,
            cache: Default::default(),
        }
    }
    pub fn get(&self, key: &K) -> Option<Rc<AotNativeExecutor>> {
        self.cache.get(key).cloned()
    }

    pub fn compile_and_insert(
        &mut self,
        key: K,
        program: &Program,
        opt_level: OptLevel,
    ) -> Rc<AotNativeExecutor> {
        let native_module = self
            .context
            .compile(program, None)
            .expect("failed to compile program");

        let registry = ProgramRegistry::new(program).expect("failed to get program registry");
        let metadata = native_module
            .metadata()
            .get::<GasMetadata>()
            .cloned()
            .expect("module should have gas metadata");

        let shared_library = {
            let object_data = module_to_object(&native_module.module(), opt_level)
                .expect("failed to convert MLIR to object");

            let shared_library_dir = Path::new(".aot-cache");
            create_dir_all(shared_library_dir).expect("failed to create shared library directory");
            let shared_library_name = format!("lib{key}{SHARED_LIBRARY_EXT}");
            let shared_library_path = shared_library_dir.join(&shared_library_name);

            object_to_shared_lib(&object_data, &shared_library_path)
                .expect("failed to link object into shared library");

            unsafe {
                Library::new(shared_library_path).expect("failed to load dynamic shared library")
            }
        };

        let executor = AotNativeExecutor::new(shared_library, registry, metadata);
        let executor = Rc::new(executor);

        self.cache.insert(key, executor.clone());

        executor
    }
}
