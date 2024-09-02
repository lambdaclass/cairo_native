//! This module contains common code used by all integration tests, which use proptest to compare various outputs based on the inputs
//! The general idea is to have a test for each libfunc if possible.

#![allow(dead_code)]

use cairo_lang_compiler::{
    compile_prepared_db, db::RootDatabase, project::setup_project, CompilerConfig,
};
use cairo_lang_filesystem::db::init_dev_corelib;
use cairo_lang_runner::{
    Arg, RunResultStarknet, RunResultValue, RunnerError, SierraCasmRunner, StarknetState,
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        ConcreteType,
    },
    ids::{ConcreteTypeId, FunctionId},
    program::Program,
    program_registry::ProgramRegistry,
};
use cairo_lang_sierra_generator::replace_ids::DebugReplacer;
use cairo_lang_starknet::{
    compile::compile_contract_in_prepared_db, contract::get_contracts_info, starknet_plugin_suite,
};
use cairo_lang_starknet_classes::{
    casm_contract_class::CasmContractClass, contract_class::ContractClass,
};
use cairo_native::{
    context::NativeContext,
    execution_result::{ContractExecutionResult, ExecutionResult},
    executor::JitNativeExecutor,
    starknet::{DummySyscallHandler, StarknetSyscallHandler},
    types::{
        felt252::{HALF_PRIME, PRIME},
        TypeBuilder,
    },
    utils::find_entry_point_by_idx,
    values::JitValue,
    OptLevel,
};
use cairo_vm::{
    hint_processor::cairo_1_hint_processor::hint_processor::Cairo1HintProcessor,
    types::{builtin_name::BuiltinName, layout_name::LayoutName, relocatable::MaybeRelocatable},
    vm::runners::cairo_runner::{CairoArg, CairoRunner, RunResources},
};
use itertools::Itertools;
use lambdaworks_math::{
    field::{
        element::FieldElement, fields::montgomery_backed_prime_fields::MontgomeryBackendPrimeField,
    },
    unsigned_integer::element::UnsignedInteger,
};
use num_bigint::{BigInt, Sign};
use proptest::{strategy::Strategy, test_runner::TestCaseError};
use starknet_types_core::felt::Felt;
use std::{collections::HashMap, env::var, fs, ops::Neg, path::Path};

#[allow(unused_macros)]
macro_rules! load_cairo {
    ( $( $program:tt )+ ) => {
        $crate::common::load_cairo_str(stringify!($($program)+))
    };
}

#[allow(unused_imports)]
pub(crate) use load_cairo;
use num_traits::ToPrimitive;

pub const DEFAULT_GAS: u64 = u64::MAX;

// Parse numeric string into felt, wrapping negatives around the prime modulo.
pub fn felt(value: &str) -> [u32; 8] {
    let value = value.parse::<BigInt>().unwrap();
    let value = match value.sign() {
        Sign::Minus => &*PRIME - value.neg().to_biguint().unwrap(),
        _ => value.to_biguint().unwrap(),
    };

    let mut u32_digits = value.to_u32_digits();
    u32_digits.resize(8, 0);
    u32_digits.try_into().unwrap()
}

/// Parse any time that can be a bigint to a felt that can be used in the cairo-native input.
pub fn feltn(value: impl Into<BigInt>) -> [u32; 8] {
    let value: BigInt = value.into();
    let value = match value.sign() {
        Sign::Minus => &*PRIME - value.neg().to_biguint().unwrap(),
        _ => value.to_biguint().unwrap(),
    };

    let mut u32_digits = value.to_u32_digits();
    u32_digits.resize(8, 0);
    u32_digits.try_into().unwrap()
}

/// Converts a casm variant to sierra.
pub const fn casm_variant_to_sierra(idx: i64, num_variants: i64) -> i64 {
    num_variants - 1 - (idx >> 1)
}

pub fn get_run_result(r: &RunResultValue) -> Vec<String> {
    match r {
        RunResultValue::Success(x) | RunResultValue::Panic(x) => {
            x.iter().map(ToString::to_string).collect()
        }
    }
}

pub fn load_cairo_str(program_str: &str) -> (String, Program, SierraCasmRunner) {
    let mut program_file = tempfile::Builder::new()
        .prefix("test_")
        .suffix(".cairo")
        .tempfile()
        .unwrap();
    fs::write(&mut program_file, program_str).unwrap();

    let mut db = RootDatabase::default();
    init_dev_corelib(
        &mut db,
        Path::new(&var("CARGO_MANIFEST_DIR").unwrap()).join("corelib/src"),
    );
    let main_crate_ids = setup_project(&mut db, program_file.path()).unwrap();
    let sierra_program_with_dbg = compile_prepared_db(
        &db,
        main_crate_ids.clone(),
        CompilerConfig {
            replace_ids: true,
            ..Default::default()
        },
    )
    .unwrap();
    let program = sierra_program_with_dbg.program;

    let module_name = program_file.path().with_extension("");
    let module_name = module_name.file_name().unwrap().to_str().unwrap();

    let replacer = DebugReplacer { db: &db };
    let contracts_info = get_contracts_info(&db, main_crate_ids, &replacer).unwrap();

    let runner = SierraCasmRunner::new(
        program.clone(),
        Some(Default::default()),
        contracts_info,
        None,
    )
    .unwrap();

    (module_name.to_string(), program, runner)
}

pub fn load_cairo_path(program_path: &str) -> (String, Program, SierraCasmRunner) {
    let program_file = Path::new(program_path);

    let mut db = RootDatabase::default();
    init_dev_corelib(
        &mut db,
        Path::new(&var("CARGO_MANIFEST_DIR").unwrap()).join("corelib/src"),
    );
    let main_crate_ids = setup_project(&mut db, program_file).unwrap();
    let sierra_program_with_dbg = compile_prepared_db(
        &db,
        main_crate_ids.clone(),
        CompilerConfig {
            replace_ids: true,
            ..Default::default()
        },
    )
    .unwrap();
    let program = sierra_program_with_dbg.program;

    let module_name = program_file.with_extension("");
    let module_name = module_name.file_name().unwrap().to_str().unwrap();

    let replacer = DebugReplacer { db: &db };
    let contracts_info = get_contracts_info(&db, main_crate_ids, &replacer).unwrap();

    let runner = SierraCasmRunner::new(
        program.clone(),
        Some(Default::default()),
        contracts_info,
        None,
    )
    .unwrap();

    (module_name.to_string(), program, runner)
}

/// Compiles a cairo starknet contract from the given path
pub fn load_cairo_contract_path(path: &str) -> ContractClass {
    let mut db = RootDatabase::builder()
        .detect_corelib()
        .with_plugin_suite(starknet_plugin_suite())
        .build()
        .expect("failed to build database");

    let main_crate_ids = setup_project(&mut db, Path::new(&path))
        .expect("path should be a valid cairo project or file");

    compile_contract_in_prepared_db(
        &db,
        None,
        main_crate_ids.clone(),
        CompilerConfig {
            replace_ids: true,
            ..Default::default()
        },
    )
    .expect("path should contain a single valid contract")
}

pub fn run_native_program(
    program: &(String, Program, SierraCasmRunner),
    entry_point: &str,
    args: &[JitValue],
    gas: Option<u128>,
    syscall_handler: Option<impl StarknetSyscallHandler>,
) -> ExecutionResult {
    let entry_point = format!("{0}::{0}::{1}", program.0, entry_point);
    let program = &program.1;

    let entry_point_id = &program
        .funcs
        .iter()
        .find(|x| x.id.debug_name.as_deref() == Some(&entry_point))
        .expect("Test program entry point not found.")
        .id;

    let context = NativeContext::new();

    let module = context
        .compile(program)
        .expect("Could not compile test program to MLIR.");

    assert!(
        module.module().as_operation().verify(),
        "Test program generated invalid MLIR:\n{}",
        module.module().as_operation()
    );

    // FIXME: There are some bugs with non-zero LLVM optimization levels.
    let executor = JitNativeExecutor::from_native_module(module, OptLevel::None);
    match syscall_handler {
        Some(syscall_handler) => executor
            .invoke_dynamic_with_syscall_handler(entry_point_id, args, gas, syscall_handler)
            .unwrap(),
        None => executor.invoke_dynamic(entry_point_id, args, gas).unwrap(),
    }
}

/// Runs the program on the cairo-vm
pub fn run_vm_program(
    program: &(String, Program, SierraCasmRunner),
    entry_point: &str,
    args: &[Arg],
    gas: Option<usize>,
) -> Result<RunResultStarknet, RunnerError> {
    let runner = &program.2;
    runner.run_function_with_starknet_context(
        runner.find_function(entry_point).unwrap(),
        args,
        gas,
        StarknetState::default(),
    )
}

/// Runs the contract on the cairo-vm
pub fn run_vm_contract(
    cairo_contract: &ContractClass,
    entrypoint: usize,
    args: &[Felt],
) -> Vec<Felt> {
    let args = args
        .iter()
        .map(|arg| MaybeRelocatable::Int(*arg))
        .collect_vec();

    let contract =
        CasmContractClass::from_contract_class(cairo_contract.clone(), false, usize::MAX)
            .expect("failed to compile sierra contract to casm");

    let program = contract
        .clone()
        .try_into()
        .expect("failed to extract program from casm contract");

    // Initialize runner and builtins
    let mut runner = CairoRunner::new(&program, LayoutName::all_cairo, false, false)
        .expect("failed to build runner");

    let program_builtins = contract
        .entry_points_by_type
        .external
        .iter()
        .find(|e| e.offset == entrypoint)
        .expect("given entrypoint index should exist")
        .builtins
        .iter()
        .map(|s| BuiltinName::from_str(s).expect("invalid builtin name"))
        .collect_vec();
    runner
        .initialize_function_runner_cairo_1(&program_builtins)
        .expect("failed to initialize runner");

    // Initialize implicit Args
    let builtins = runner.get_program_builtins();
    let mut implicit_args: Vec<MaybeRelocatable> = runner
        .vm
        .get_builtin_runners()
        .iter()
        .filter(|b| builtins.contains(&b.name()))
        .flat_map(|b| b.initial_stack())
        .collect();
    let initial_gas = MaybeRelocatable::from(usize::MAX);
    implicit_args.extend([initial_gas]);
    let syscall_segment = MaybeRelocatable::from(runner.vm.add_memory_segment());
    implicit_args.extend([syscall_segment]);

    // Load builtin costs
    let builtin_costs: Vec<MaybeRelocatable> =
        vec![0.into(), 0.into(), 0.into(), 0.into(), 0.into()];
    let builtin_costs_ptr = runner.vm.add_memory_segment();

    runner
        .vm
        .load_data(builtin_costs_ptr, &builtin_costs)
        .expect("failed to load builtin costs data to vm");

    // Load extra data
    let core_program_end_ptr = (runner.program_base.expect("program base is missing")
        + runner.get_program().data_len())
    .expect("memory pointer overflowed");

    let program_extra_data: Vec<MaybeRelocatable> =
        vec![0x208B7FFF7FFF7FFE.into(), builtin_costs_ptr.into()];
    runner
        .vm
        .load_data(core_program_end_ptr, &program_extra_data)
        .expect("failed to load extra data to vm");

    // Load calldata
    let calldata_start = runner.vm.add_memory_segment();
    let calldata_end = runner
        .vm
        .load_data(calldata_start, &args.to_vec())
        .expect("failed to load calldata to vm");

    // Create entrypoint_args
    let mut entrypoint_args: Vec<CairoArg> = implicit_args
        .iter()
        .map(|m| CairoArg::from(m.clone()))
        .collect();
    entrypoint_args.extend([
        MaybeRelocatable::from(calldata_start).into(),
        MaybeRelocatable::from(calldata_end).into(),
    ]);
    let entrypoint_args: Vec<&CairoArg> = entrypoint_args.iter().collect();

    // Run contract entrypoint
    let mut hint_processor =
        Cairo1HintProcessor::new(&contract.hints, RunResources::default(), false);
    runner
        .run_from_entrypoint(
            entrypoint,
            &entrypoint_args,
            true,
            Some(runner.get_program().data_len() + program_extra_data.len()),
            &mut hint_processor,
        )
        .expect("failed to execute contract");

    // Extract return values
    let return_values = runner
        .vm
        .get_return_values(5)
        .expect("failed to extract return values");
    let retdata_start = return_values[3]
        .get_relocatable()
        .expect("failed to get return data start");
    let retdata_end = return_values[4]
        .get_relocatable()
        .expect("failed to get return data end");

    runner
        .vm
        .get_integer_range(
            retdata_start,
            (retdata_end - retdata_start).expect("return data length should not be negative"),
        )
        .expect("failed to access vm memory")
        .iter()
        .map(|c| c.clone().into_owned())
        .collect_vec()
}

#[track_caller]
pub fn compare_inputless_program(
    program_path: &str,
    entry_point: Option<&str>,
    custom_stack: Option<usize>,
) {
    let program: (String, Program, SierraCasmRunner) = load_cairo_path(program_path);

    let result_vm = run_vm_program(&program, "main", &[], Some(DEFAULT_GAS as usize)).unwrap();
    let (program, result_native) = match custom_stack {
        Some(custom_stack) => std::thread::Builder::new()
            .name("TEST RUNNER".to_string())
            .stack_size(custom_stack)
            .spawn({
                let entry_point = entry_point.unwrap_or("main").to_string();
                move || {
                    let result = run_native_program(
                        &program,
                        &entry_point,
                        &[],
                        Some(DEFAULT_GAS as u128),
                        Option::<DummySyscallHandler>::None,
                    );
                    (program, result)
                }
            })
            .unwrap()
            .join()
            .unwrap(),
        None => {
            let result = run_native_program(
                &program,
                "main",
                &[],
                Some(DEFAULT_GAS as u128),
                Option::<DummySyscallHandler>::None,
            );
            (program, result)
        }
    };

    compare_outputs(
        &program.1,
        &program.2.find_function("main").unwrap().id,
        &result_vm,
        &result_native,
    )
    .expect("compare error");
}

/// Runs the program using cairo-native JIT.
pub fn run_native_starknet_contract(
    sierra_program: &Program,
    entry_point_function_idx: usize,
    args: &[Felt],
    handler: impl StarknetSyscallHandler,
) -> ContractExecutionResult {
    let native_context = NativeContext::new();

    let native_program = native_context.compile(sierra_program).unwrap();

    let entry_point_fn = find_entry_point_by_idx(sierra_program, entry_point_function_idx).unwrap();
    let entry_point_id = &entry_point_fn.id;

    let native_executor = JitNativeExecutor::from_native_module(native_program, Default::default());
    native_executor
        .invoke_contract_dynamic(entry_point_id, args, u128::MAX.into(), handler)
        .expect("failed to execute the given contract")
}

/// Given the result of the cairo-vm and cairo-native of the same program, it compares
/// the results automatically, triggering a proptest assert if there is a mismatch.
///
/// Left of report of the assert is the cairo vm result, right side is cairo native
#[track_caller]
pub fn compare_outputs(
    program: &Program,
    entry_point: &FunctionId,
    vm_result: &RunResultStarknet,
    native_result: &ExecutionResult,
) -> Result<(), TestCaseError> {
    let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(program).unwrap();
    let function = registry.get_function(entry_point).unwrap();

    fn map_vm_sizes(
        size_cache: &mut HashMap<ConcreteTypeId, usize>,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        ty: &ConcreteTypeId,
    ) -> usize {
        match size_cache.get(ty) {
            Some(&type_size) => type_size,
            None => {
                let type_size = match registry.get_type(ty).unwrap() {
                    CoreTypeConcrete::Array(_) | CoreTypeConcrete::EcPoint(_) => 2,
                    CoreTypeConcrete::Felt252(_)
                    | CoreTypeConcrete::Uint128(_)
                    | CoreTypeConcrete::Uint64(_)
                    | CoreTypeConcrete::Uint32(_)
                    | CoreTypeConcrete::Uint16(_)
                    | CoreTypeConcrete::Uint8(_)
                    | CoreTypeConcrete::Sint128(_)
                    | CoreTypeConcrete::Sint64(_)
                    | CoreTypeConcrete::Sint32(_)
                    | CoreTypeConcrete::Sint16(_)
                    | CoreTypeConcrete::Sint8(_)
                    | CoreTypeConcrete::Box(_)
                    | CoreTypeConcrete::Nullable(_) => 1,
                    CoreTypeConcrete::Enum(info) => {
                        1 + info
                            .variants
                            .iter()
                            .map(|variant_ty| map_vm_sizes(size_cache, registry, variant_ty))
                            .max()
                            .unwrap_or_default()
                    }
                    CoreTypeConcrete::Struct(info) => info
                        .members
                        .iter()
                        .map(|member_ty| map_vm_sizes(size_cache, registry, member_ty))
                        .sum(),
                    CoreTypeConcrete::NonZero(info) => map_vm_sizes(size_cache, registry, &info.ty),
                    CoreTypeConcrete::EcState(_) => 4,
                    CoreTypeConcrete::Snapshot(info) => {
                        map_vm_sizes(size_cache, registry, &info.ty)
                    }
                    CoreTypeConcrete::SquashedFelt252Dict(_) => 2,
                    x => todo!("vm size not yet implemented: {:?}", x.info()),
                };
                size_cache.insert(ty.clone(), type_size);

                type_size
            }
        }
    }

    fn map_vm_values(
        size_cache: &mut HashMap<ConcreteTypeId, usize>,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        memory: &[Option<Felt>],
        mut values: &[Felt],
        ty: &ConcreteTypeId,
    ) -> JitValue {
        match registry.get_type(ty).unwrap() {
            CoreTypeConcrete::Array(info) => {
                assert_eq!(values.len(), 2);
                let since_ptr = values[0].to_usize().unwrap();
                let until_ptr = values[1].to_usize().unwrap();

                let total_len = until_ptr - since_ptr;
                let elem_size = map_vm_sizes(size_cache, registry, &info.ty);
                assert_eq!(total_len % elem_size, 0);

                JitValue::Array(
                    memory[since_ptr..until_ptr]
                        .chunks(elem_size)
                        .map(|data| data.iter().cloned().map(Option::unwrap).collect::<Vec<_>>())
                        .map(|data| map_vm_values(size_cache, registry, memory, &data, &info.ty))
                        .collect(),
                )
            }
            CoreTypeConcrete::Felt252(_) => {
                JitValue::Felt252(Felt::from_bytes_le(&values[0].to_bytes_le()))
            }
            CoreTypeConcrete::Uint128(_) => JitValue::Uint128(values[0].to_u128().unwrap()),
            CoreTypeConcrete::Uint64(_) => JitValue::Uint64(values[0].to_u64().unwrap()),
            CoreTypeConcrete::Uint32(_) => JitValue::Uint32(values[0].to_u32().unwrap()),
            CoreTypeConcrete::Uint16(_) => JitValue::Uint16(values[0].to_u16().unwrap()),
            CoreTypeConcrete::Uint8(_) => JitValue::Uint8(values[0].to_u8().unwrap()),
            CoreTypeConcrete::Sint128(_) => {
                JitValue::Sint128(if values[0].to_bigint() >= *HALF_PRIME {
                    -(&*PRIME - &values[0].to_biguint()).to_i128().unwrap()
                } else {
                    values[0].to_biguint().to_i128().unwrap()
                })
            }
            CoreTypeConcrete::Sint64(_) => {
                JitValue::Sint64(if values[0].to_bigint() >= *HALF_PRIME {
                    -(&*PRIME - &values[0].to_biguint()).to_i64().unwrap()
                } else {
                    values[0].to_biguint().to_i64().unwrap()
                })
            }
            CoreTypeConcrete::Sint32(_) => {
                JitValue::Sint32(if values[0].to_bigint() >= *HALF_PRIME {
                    -(&*PRIME - &values[0].to_biguint()).to_i32().unwrap()
                } else {
                    values[0].to_biguint().to_i32().unwrap()
                })
            }
            CoreTypeConcrete::Sint16(_) => {
                JitValue::Sint16(if values[0].to_bigint() >= *HALF_PRIME {
                    -(&*PRIME - &values[0].to_biguint()).to_i16().unwrap()
                } else {
                    values[0].to_biguint().to_i16().unwrap()
                })
            }
            CoreTypeConcrete::Sint8(_) => {
                JitValue::Sint8(if values[0].to_bigint() >= *HALF_PRIME {
                    -(&*PRIME - &values[0].to_biguint()).to_i8().unwrap()
                } else {
                    values[0].to_biguint().to_i8().unwrap()
                })
            }
            CoreTypeConcrete::Enum(info) => {
                let enum_size = map_vm_sizes(size_cache, registry, ty);
                assert_eq!(values.len(), enum_size);

                let (tag, data);
                (tag, values) = values.split_first().unwrap();

                let mut tag = tag.to_usize().unwrap();
                if info.variants.len() > 2 {
                    tag = info.variants.len() - ((tag + 1) >> 1);
                }
                assert!(tag <= info.variants.len());
                data = &values[enum_size - size_cache[&info.variants[tag]] - 1..];

                JitValue::Enum {
                    tag,
                    value: Box::new(map_vm_values(
                        size_cache,
                        registry,
                        memory,
                        data,
                        &info.variants[tag],
                    )),
                    debug_name: ty.debug_name.as_deref().map(String::from),
                }
            }
            CoreTypeConcrete::Struct(info) => JitValue::Struct {
                fields: info
                    .members
                    .iter()
                    .map(|member_ty| {
                        let data;
                        (data, values) =
                            values.split_at(map_vm_sizes(size_cache, registry, member_ty));

                        map_vm_values(size_cache, registry, memory, data, member_ty)
                    })
                    .collect(),
                debug_name: ty.debug_name.as_deref().map(String::from),
            },
            CoreTypeConcrete::SquashedFelt252Dict(info) => JitValue::Felt252Dict {
                value: (values[0].to_usize().unwrap()..values[1].to_usize().unwrap())
                    .step_by(3)
                    .map(|index| {
                        (
                            Felt::from_bytes_le(&memory[index].unwrap().to_bytes_le()),
                            match &info.info.long_id.generic_args[0] {
                                cairo_lang_sierra::program::GenericArg::Type(ty) => map_vm_values(
                                    size_cache,
                                    registry,
                                    memory,
                                    &[memory[index + 2].unwrap()],
                                    ty,
                                ),
                                _ => unimplemented!("unsupported dict value type"),
                            },
                        )
                    })
                    .collect(),
                debug_name: ty.debug_name.as_deref().map(String::from),
            },
            CoreTypeConcrete::Snapshot(info) => {
                map_vm_values(size_cache, registry, memory, values, &info.ty)
            }
            CoreTypeConcrete::Nullable(info) => {
                assert_eq!(values.len(), 1);

                let ty_size = map_vm_sizes(size_cache, registry, &info.ty);
                match values[0].to_usize().unwrap() {
                    0 => JitValue::Null,
                    ptr if ty_size == 0 => {
                        assert_eq!(ptr, 1);
                        map_vm_values(size_cache, registry, memory, &[], &info.ty)
                    }
                    ptr => map_vm_values(
                        size_cache,
                        registry,
                        memory,
                        &memory[ptr..ptr + ty_size]
                            .iter()
                            .cloned()
                            .map(Option::unwrap)
                            .collect::<Vec<_>>(),
                        &info.ty,
                    ),
                }
            }
            CoreTypeConcrete::Box(info) => {
                assert_eq!(values.len(), 1);

                let ty_size = map_vm_sizes(size_cache, registry, &info.ty);
                match values[0].to_usize().unwrap() {
                    ptr if ty_size == 0 => {
                        assert_eq!(ptr, 1);
                        map_vm_values(size_cache, registry, memory, &[], &info.ty)
                    }
                    ptr => map_vm_values(
                        size_cache,
                        registry,
                        memory,
                        &memory[ptr..ptr + ty_size]
                            .iter()
                            .cloned()
                            .map(Option::unwrap)
                            .collect::<Vec<_>>(),
                        &info.ty,
                    ),
                }
            }
            CoreTypeConcrete::NonZero(info) => {
                map_vm_values(size_cache, registry, memory, values, &info.ty)
            }
            CoreTypeConcrete::EcPoint(_) => {
                assert_eq!(values.len(), 2);

                JitValue::EcPoint(
                    Felt::from_bytes_le(&values[0].to_bytes_le()),
                    Felt::from_bytes_le(&values[1].to_bytes_le()),
                )
            }
            CoreTypeConcrete::EcState(_) => {
                assert_eq!(values.len(), 4);

                JitValue::EcState(
                    Felt::from_bytes_le(&values[0].to_bytes_le()),
                    Felt::from_bytes_le(&values[1].to_bytes_le()),
                    Felt::from_bytes_le(&values[2].to_bytes_le()),
                    Felt::from_bytes_le(&values[3].to_bytes_le()),
                )
            }
            CoreTypeConcrete::Bytes31(_) => {
                let mut bytes = values[0].to_bytes_le().to_vec();
                bytes.pop();
                JitValue::Bytes31(bytes.try_into().unwrap())
            }
            CoreTypeConcrete::Coupon(_) => todo!(),
            CoreTypeConcrete::Bitwise(_) => unreachable!(),
            CoreTypeConcrete::Const(_) => unreachable!(),
            CoreTypeConcrete::EcOp(_) => unreachable!(),
            CoreTypeConcrete::GasBuiltin(_) => unreachable!(),
            CoreTypeConcrete::BuiltinCosts(_) => unreachable!(),
            CoreTypeConcrete::RangeCheck(_) => unreachable!(),
            CoreTypeConcrete::Pedersen(_) => unreachable!(),
            CoreTypeConcrete::Poseidon(_) => unreachable!(),
            CoreTypeConcrete::SegmentArena(_) => unreachable!(),
            CoreTypeConcrete::BoundedInt(_) => unreachable!(),
            x => {
                todo!("vm value not yet implemented: {:?}", x.info())
            }
        }
    }

    let mut size_cache = HashMap::new();
    let ty = function.signature.ret_types.last();
    let is_builtin = ty.map_or(false, |ty| registry.get_type(ty).unwrap().is_builtin());
    let returns_panic = ty.map_or(false, |ty| {
        ty.debug_name
            .as_ref()
            .map(|x| x.starts_with("core::panics::PanicResult"))
            .unwrap_or(false)
    });
    assert_eq!(
        vm_result.gas_counter.unwrap_or_else(|| Felt::from(0)),
        Felt::from(native_result.remaining_gas.unwrap_or(0)),
    );

    let vm_result = match &vm_result.value {
        RunResultValue::Success(values) if !values.is_empty() | returns_panic => {
            if returns_panic {
                let inner_ty = match registry.get_type(ty.unwrap())? {
                    CoreTypeConcrete::Enum(info) => &info.variants[0],
                    _ => unreachable!(),
                };
                JitValue::Enum {
                    tag: 0,
                    value: Box::new(map_vm_values(
                        &mut size_cache,
                        &registry,
                        &vm_result.memory,
                        values,
                        inner_ty,
                    )),
                    debug_name: None,
                }
            } else if !is_builtin {
                map_vm_values(
                    &mut size_cache,
                    &registry,
                    &vm_result.memory,
                    values,
                    ty.unwrap(),
                )
            } else {
                JitValue::Struct {
                    fields: Vec::new(),
                    debug_name: None,
                }
            }
        }
        RunResultValue::Panic(values) => JitValue::Enum {
            tag: 1,
            value: Box::new(JitValue::Struct {
                fields: vec![
                    JitValue::Struct {
                        fields: Vec::new(),
                        debug_name: None,
                    },
                    JitValue::Array(
                        values
                            .iter()
                            .map(|value| Felt::from_bytes_le(&value.to_bytes_le()))
                            .map(JitValue::Felt252)
                            .collect(),
                    ),
                ],
                debug_name: None,
            }),
            debug_name: None,
        },
        _ => JitValue::Struct {
            fields: vec![],
            debug_name: None,
        },
    };

    pretty_assertions_sorted::assert_eq!(native_result.return_value, vm_result);
    Ok(())
}

pub const FIELD_HIGH: u128 = (1 << 123) + (17 << 64); // this is equal to 10633823966279327296825105735305134080
pub const FIELD_LOW: u128 = 1;

/// Returns a [`Strategy`] that generates any valid Felt
pub fn any_felt() -> impl Strategy<Value = Felt> {
    use proptest::prelude::*;

    (0..=FIELD_HIGH)
        // turn range into `impl Strategy`
        .prop_map(|x| x)
        // choose second 128-bit limb capped by first one
        .prop_flat_map(|high| {
            let low = if high == FIELD_HIGH {
                (0..FIELD_LOW).prop_map(|x| x).sboxed()
            } else {
                any::<u128>().sboxed()
            };
            (Just(high), low)
        })
        // turn (u128, u128) into limbs array and then into Felt
        .prop_map(|(high, low)| {
            let limbs = [
                (high >> 64) as u64,
                (high & ((1 << 64) - 1)) as u64,
                (low >> 64) as u64,
                (low & ((1 << 64) - 1)) as u64,
            ];
            FieldElement::new(UnsignedInteger::from_limbs(limbs))
        })
        .prop_map(|value: FieldElement<MontgomeryBackendPrimeField<_, 4>>| {
            Felt::from_bytes_be(&value.to_bytes_be())
        })
}

/// Returns a [`Strategy`] that generates any nonzero Felt
pub fn nonzero_felt() -> impl Strategy<Value = Felt> {
    any_felt().prop_filter("is zero", |x| x != &Felt::ZERO)
}
