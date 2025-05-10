use std::{fs::File, io::Write, path::PathBuf};

use bincode::enc::write::Writer;
use cairo_lang_starknet_classes::casm_contract_class::CasmContractClass;
use cairo_vm::{
    cairo_run::{write_encoded_memory, write_encoded_trace},
    hint_processor::cairo_1_hint_processor::hint_processor::Cairo1HintProcessor,
    types::{
        builtin_name::BuiltinName, layout_name::LayoutName, program::Program,
        relocatable::MaybeRelocatable,
    },
    vm::runners::cairo_runner::{CairoArg, CairoRunner, RunResources},
};
use clap::Parser;
use starknet_types_core::felt::Felt;

#[derive(Debug, Parser)]
struct Args {
    contract_path: PathBuf,
    memory_path: PathBuf,
    trace_path: PathBuf,
}

pub fn main() {
    let cli_args = Args::parse();
    let calldata_args = [MaybeRelocatable::from(10)];
    let expected_retdata = [Felt::from(89)];

    let contract_file = File::open(cli_args.contract_path).expect("failed to open contract path");
    let contract: CasmContractClass =
        serde_json::from_reader(contract_file).expect("failed to parse contract file");

    let program = Program::try_from(contract.clone())
        .expect("failed to build vm program from contract class");

    let mut runner =
        CairoRunner::new(&program, LayoutName::all_cairo, None, false, true, false).unwrap();

    let program_builtins = contract
        .entry_points_by_type
        .external
        .iter()
        .find(|e| e.offset == 0)
        .unwrap()
        .builtins
        .iter()
        .map(|s| BuiltinName::from_str(s).expect("Invalid builtin name"))
        .collect::<Vec<_>>();
    runner
        .initialize_function_runner_cairo_1(&program_builtins)
        .unwrap();

    let syscall_segment = MaybeRelocatable::from(runner.vm.add_memory_segment());

    let builtins = runner.get_program_builtins();

    let builtin_segment: Vec<MaybeRelocatable> = runner
        .vm
        .get_builtin_runners()
        .iter()
        .filter(|b| builtins.contains(&b.name()))
        .flat_map(|b| b.initial_stack())
        .collect();

    let initial_gas = MaybeRelocatable::from(usize::MAX);
    println!("Starting Gas: {}", &initial_gas);

    let mut implicit_args = builtin_segment;
    implicit_args.extend([initial_gas]);
    implicit_args.extend([syscall_segment]);

    let builtin_costs: Vec<MaybeRelocatable> =
        vec![0.into(), 0.into(), 0.into(), 0.into(), 0.into()];
    let builtin_costs_ptr = runner.vm.add_memory_segment();
    runner
        .vm
        .load_data(builtin_costs_ptr, &builtin_costs)
        .unwrap();

    let core_program_end_ptr =
        (runner.program_base.unwrap() + runner.get_program().data_len()).unwrap();
    let program_extra_data: Vec<MaybeRelocatable> =
        vec![0x208B7FFF7FFF7FFE.into(), builtin_costs_ptr.into()];
    runner
        .vm
        .load_data(core_program_end_ptr, &program_extra_data)
        .unwrap();

    let calldata_start = runner.vm.add_memory_segment();
    let calldata_end = runner.vm.load_data(calldata_start, &calldata_args).unwrap();

    let mut entrypoint_args: Vec<CairoArg> = implicit_args
        .iter()
        .map(|m| CairoArg::from(m.clone()))
        .collect();
    entrypoint_args.extend([
        MaybeRelocatable::from(calldata_start).into(),
        MaybeRelocatable::from(calldata_end).into(),
    ]);
    let entrypoint_args: Vec<&CairoArg> = entrypoint_args.iter().collect();

    let mut hint_processor =
        Cairo1HintProcessor::new(&contract.hints, RunResources::new(621), false);

    runner
        .run_from_entrypoint(
            0,
            &entrypoint_args,
            true,
            Some(runner.get_program().data_len() + program_extra_data.len()),
            &mut hint_processor,
        )
        .expect("failed to execute contract");

    let return_values = runner.vm.get_return_values(5).unwrap();
    let final_gas = return_values[0].get_int().unwrap();
    let retdata_start = return_values[3].get_relocatable().unwrap();
    let retdata_end = return_values[4].get_relocatable().unwrap();
    let retdata: Vec<Felt> = runner
        .vm
        .get_integer_range(retdata_start, (retdata_end - retdata_start).unwrap())
        .unwrap()
        .iter()
        .map(|c| c.clone().into_owned())
        .collect();

    println!("Final Gas: {}", final_gas);

    assert_eq!(retdata, expected_retdata);

    runner.relocate(true).expect("failed to relocate trace");

    let trace_file = File::create(cli_args.trace_path).expect("failed to create trace file");
    let mut trace_writer = FileWriter::new(trace_file);
    write_encoded_trace(
        &runner.relocated_trace.expect("trace should exist"),
        &mut trace_writer,
    )
    .expect("failed to write trace");

    let memory_file = File::create(cli_args.memory_path).expect("failed to create memory file");
    let mut memory_writer = FileWriter::new(memory_file);
    write_encoded_memory(&runner.relocated_memory, &mut memory_writer)
        .expect("failed to write memory");
}

struct FileWriter {
    buf_writer: File,
    bytes_written: usize,
}

impl Writer for FileWriter {
    fn write(&mut self, bytes: &[u8]) -> Result<(), bincode::error::EncodeError> {
        self.buf_writer
            .write_all(bytes)
            .map_err(|e| bincode::error::EncodeError::Io {
                inner: e,
                index: self.bytes_written,
            })?;

        self.bytes_written += bytes.len();

        Ok(())
    }
}

impl FileWriter {
    fn new(buf_writer: File) -> Self {
        Self {
            buf_writer,
            bytes_written: 0,
        }
    }
}
