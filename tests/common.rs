////! This module contains common code used by all integration tests, which use proptest to compare various outputs based on the inputs
//! This module contains common code used by all integration tests, which use proptest to compare various outputs based on the inputs
////! The general idea is to have a test for each libfunc if possible.
//! The general idea is to have a test for each libfunc if possible.
//

//#![allow(dead_code)]
#![allow(dead_code)]
//

//use cairo_lang_compiler::{
use cairo_lang_compiler::{
//    compile_prepared_db, db::RootDatabase, project::setup_project, CompilerConfig,
    compile_prepared_db, db::RootDatabase, project::setup_project, CompilerConfig,
//};
};
//use cairo_lang_filesystem::db::init_dev_corelib;
use cairo_lang_filesystem::db::init_dev_corelib;
//use cairo_lang_runner::{
use cairo_lang_runner::{
//    Arg, RunResultStarknet, RunResultValue, RunnerError, SierraCasmRunner, StarknetState,
    Arg, RunResultStarknet, RunResultValue, RunnerError, SierraCasmRunner, StarknetState,
//};
};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
//        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
//        ConcreteType,
        ConcreteType,
//    },
    },
//    ids::{ConcreteTypeId, FunctionId},
    ids::{ConcreteTypeId, FunctionId},
//    program::Program,
    program::Program,
//    program_registry::ProgramRegistry,
    program_registry::ProgramRegistry,
//};
};
//use cairo_lang_sierra_generator::replace_ids::DebugReplacer;
use cairo_lang_sierra_generator::replace_ids::DebugReplacer;
//use cairo_lang_starknet::contract::get_contracts_info;
use cairo_lang_starknet::contract::get_contracts_info;
//use cairo_native::{
use cairo_native::{
//    context::NativeContext,
    context::NativeContext,
//    execution_result::{ContractExecutionResult, ExecutionResult},
    execution_result::{ContractExecutionResult, ExecutionResult},
//    executor::JitNativeExecutor,
    executor::JitNativeExecutor,
//    starknet::{DummySyscallHandler, StarknetSyscallHandler},
    starknet::{DummySyscallHandler, StarknetSyscallHandler},
//    types::felt252::{HALF_PRIME, PRIME},
    types::felt252::{HALF_PRIME, PRIME},
//    utils::find_entry_point_by_idx,
    utils::find_entry_point_by_idx,
//    values::JitValue,
    values::JitValue,
//    OptLevel,
    OptLevel,
//};
};
//use lambdaworks_math::{
use lambdaworks_math::{
//    field::{
    field::{
//        element::FieldElement, fields::montgomery_backed_prime_fields::MontgomeryBackendPrimeField,
        element::FieldElement, fields::montgomery_backed_prime_fields::MontgomeryBackendPrimeField,
//    },
    },
//    unsigned_integer::element::UnsignedInteger,
    unsigned_integer::element::UnsignedInteger,
//};
};
//use num_bigint::{BigInt, Sign};
use num_bigint::{BigInt, Sign};
//use proptest::{strategy::Strategy, test_runner::TestCaseError};
use proptest::{strategy::Strategy, test_runner::TestCaseError};
//use starknet_types_core::felt::Felt;
use starknet_types_core::felt::Felt;
//use std::{collections::HashMap, env::var, fs, ops::Neg, path::Path};
use std::{collections::HashMap, env::var, fs, ops::Neg, path::Path};
//

//#[allow(unused_macros)]
#[allow(unused_macros)]
//macro_rules! load_cairo {
macro_rules! load_cairo {
//    ( $( $program:tt )+ ) => {
    ( $( $program:tt )+ ) => {
//        $crate::common::load_cairo_str(stringify!($($program)+))
        $crate::common::load_cairo_str(stringify!($($program)+))
//    };
    };
//}
}
//

//use cairo_felt::Felt252;
use cairo_felt::Felt252;
//#[allow(unused_imports)]
#[allow(unused_imports)]
//pub(crate) use load_cairo;
pub(crate) use load_cairo;
//use num_traits::ToPrimitive;
use num_traits::ToPrimitive;
//

//pub const DEFAULT_GAS: u64 = u64::MAX;
pub const DEFAULT_GAS: u64 = u64::MAX;
//

//// Parse numeric string into felt, wrapping negatives around the prime modulo.
// Parse numeric string into felt, wrapping negatives around the prime modulo.
//pub fn felt(value: &str) -> [u32; 8] {
pub fn felt(value: &str) -> [u32; 8] {
//    let value = value.parse::<BigInt>().unwrap();
    let value = value.parse::<BigInt>().unwrap();
//    let value = match value.sign() {
    let value = match value.sign() {
//        Sign::Minus => &*PRIME - value.neg().to_biguint().unwrap(),
        Sign::Minus => &*PRIME - value.neg().to_biguint().unwrap(),
//        _ => value.to_biguint().unwrap(),
        _ => value.to_biguint().unwrap(),
//    };
    };
//

//    let mut u32_digits = value.to_u32_digits();
    let mut u32_digits = value.to_u32_digits();
//    u32_digits.resize(8, 0);
    u32_digits.resize(8, 0);
//    u32_digits.try_into().unwrap()
    u32_digits.try_into().unwrap()
//}
}
//

///// Parse any time that can be a bigint to a felt that can be used in the cairo-native input.
/// Parse any time that can be a bigint to a felt that can be used in the cairo-native input.
//pub fn feltn(value: impl Into<BigInt>) -> [u32; 8] {
pub fn feltn(value: impl Into<BigInt>) -> [u32; 8] {
//    let value: BigInt = value.into();
    let value: BigInt = value.into();
//    let value = match value.sign() {
    let value = match value.sign() {
//        Sign::Minus => &*PRIME - value.neg().to_biguint().unwrap(),
        Sign::Minus => &*PRIME - value.neg().to_biguint().unwrap(),
//        _ => value.to_biguint().unwrap(),
        _ => value.to_biguint().unwrap(),
//    };
    };
//

//    let mut u32_digits = value.to_u32_digits();
    let mut u32_digits = value.to_u32_digits();
//    u32_digits.resize(8, 0);
    u32_digits.resize(8, 0);
//    u32_digits.try_into().unwrap()
    u32_digits.try_into().unwrap()
//}
}
//

///// Converts a casm variant to sierra.
/// Converts a casm variant to sierra.
//pub const fn casm_variant_to_sierra(idx: i64, num_variants: i64) -> i64 {
pub const fn casm_variant_to_sierra(idx: i64, num_variants: i64) -> i64 {
//    num_variants - 1 - (idx >> 1)
    num_variants - 1 - (idx >> 1)
//}
}
//

//pub fn get_run_result(r: &RunResultValue) -> Vec<String> {
pub fn get_run_result(r: &RunResultValue) -> Vec<String> {
//    match r {
    match r {
//        RunResultValue::Success(x) | RunResultValue::Panic(x) => {
        RunResultValue::Success(x) | RunResultValue::Panic(x) => {
//            x.iter().map(ToString::to_string).collect()
            x.iter().map(ToString::to_string).collect()
//        }
        }
//    }
    }
//}
}
//

//pub fn load_cairo_str(program_str: &str) -> (String, Program, SierraCasmRunner) {
pub fn load_cairo_str(program_str: &str) -> (String, Program, SierraCasmRunner) {
//    let mut program_file = tempfile::Builder::new()
    let mut program_file = tempfile::Builder::new()
//        .prefix("test_")
        .prefix("test_")
//        .suffix(".cairo")
        .suffix(".cairo")
//        .tempfile()
        .tempfile()
//        .unwrap();
        .unwrap();
//    fs::write(&mut program_file, program_str).unwrap();
    fs::write(&mut program_file, program_str).unwrap();
//

//    let mut db = RootDatabase::default();
    let mut db = RootDatabase::default();
//    init_dev_corelib(
    init_dev_corelib(
//        &mut db,
        &mut db,
//        Path::new(&var("CARGO_MANIFEST_DIR").unwrap()).join("corelib/src"),
        Path::new(&var("CARGO_MANIFEST_DIR").unwrap()).join("corelib/src"),
//    );
    );
//    let main_crate_ids = setup_project(&mut db, program_file.path()).unwrap();
    let main_crate_ids = setup_project(&mut db, program_file.path()).unwrap();
//    let program = compile_prepared_db(
    let program = compile_prepared_db(
//        &mut db,
        &mut db,
//        main_crate_ids.clone(),
        main_crate_ids.clone(),
//        CompilerConfig {
        CompilerConfig {
//            replace_ids: true,
            replace_ids: true,
//            ..Default::default()
            ..Default::default()
//        },
        },
//    )
    )
//    .unwrap();
    .unwrap();
//

//    let module_name = program_file.path().with_extension("");
    let module_name = program_file.path().with_extension("");
//    let module_name = module_name.file_name().unwrap().to_str().unwrap();
    let module_name = module_name.file_name().unwrap().to_str().unwrap();
//

//    let replacer = DebugReplacer { db: &db };
    let replacer = DebugReplacer { db: &db };
//    let contracts_info = get_contracts_info(&db, main_crate_ids, &replacer).unwrap();
    let contracts_info = get_contracts_info(&db, main_crate_ids, &replacer).unwrap();
//

//    let runner = SierraCasmRunner::new(
    let runner = SierraCasmRunner::new(
//        program.clone(),
        program.clone(),
//        Some(Default::default()),
        Some(Default::default()),
//        contracts_info,
        contracts_info,
//        None,
        None,
//    )
    )
//    .unwrap();
    .unwrap();
//

//    (module_name.to_string(), program, runner)
    (module_name.to_string(), program, runner)
//}
}
//

//pub fn load_cairo_path(program_path: &str) -> (String, Program, SierraCasmRunner) {
pub fn load_cairo_path(program_path: &str) -> (String, Program, SierraCasmRunner) {
//    let program_file = Path::new(program_path);
    let program_file = Path::new(program_path);
//

//    let mut db = RootDatabase::default();
    let mut db = RootDatabase::default();
//    init_dev_corelib(
    init_dev_corelib(
//        &mut db,
        &mut db,
//        Path::new(
        Path::new(
//            &var("CARGO_MANIFEST_DIR")
            &var("CARGO_MANIFEST_DIR")
//                .unwrap_or_else(|_| "/Users/esteve/Documents/LambdaClass/cairo_native".to_string()),
                .unwrap_or_else(|_| "/Users/esteve/Documents/LambdaClass/cairo_native".to_string()),
//        )
        )
//        .join("corelib/src"),
        .join("corelib/src"),
//    );
    );
//    let main_crate_ids = setup_project(&mut db, program_file).unwrap();
    let main_crate_ids = setup_project(&mut db, program_file).unwrap();
//    let program = compile_prepared_db(
    let program = compile_prepared_db(
//        &mut db,
        &mut db,
//        main_crate_ids.clone(),
        main_crate_ids.clone(),
//        CompilerConfig {
        CompilerConfig {
//            replace_ids: true,
            replace_ids: true,
//            ..Default::default()
            ..Default::default()
//        },
        },
//    )
    )
//    .unwrap();
    .unwrap();
//

//    let module_name = program_file.with_extension("");
    let module_name = program_file.with_extension("");
//    let module_name = module_name.file_name().unwrap().to_str().unwrap();
    let module_name = module_name.file_name().unwrap().to_str().unwrap();
//

//    let replacer = DebugReplacer { db: &db };
    let replacer = DebugReplacer { db: &db };
//    let contracts_info = get_contracts_info(&db, main_crate_ids, &replacer).unwrap();
    let contracts_info = get_contracts_info(&db, main_crate_ids, &replacer).unwrap();
//

//    let runner = SierraCasmRunner::new(
    let runner = SierraCasmRunner::new(
//        program.clone(),
        program.clone(),
//        Some(Default::default()),
        Some(Default::default()),
//        contracts_info,
        contracts_info,
//        None,
        None,
//    )
    )
//    .unwrap();
    .unwrap();
//

//    (module_name.to_string(), program, runner)
    (module_name.to_string(), program, runner)
//}
}
//

//pub fn run_native_program(
pub fn run_native_program(
//    program: &(String, Program, SierraCasmRunner),
    program: &(String, Program, SierraCasmRunner),
//    entry_point: &str,
    entry_point: &str,
//    args: &[JitValue],
    args: &[JitValue],
//    gas: Option<u128>,
    gas: Option<u128>,
//    syscall_handler: Option<impl StarknetSyscallHandler>,
    syscall_handler: Option<impl StarknetSyscallHandler>,
//) -> ExecutionResult {
) -> ExecutionResult {
//    let entry_point = format!("{0}::{0}::{1}", program.0, entry_point);
    let entry_point = format!("{0}::{0}::{1}", program.0, entry_point);
//    let program = &program.1;
    let program = &program.1;
//

//    let entry_point_id = &program
    let entry_point_id = &program
//        .funcs
        .funcs
//        .iter()
        .iter()
//        .find(|x| x.id.debug_name.as_deref() == Some(&entry_point))
        .find(|x| x.id.debug_name.as_deref() == Some(&entry_point))
//        .expect("Test program entry point not found.")
        .expect("Test program entry point not found.")
//        .id;
        .id;
//

//    let context = NativeContext::new();
    let context = NativeContext::new();
//

//    let module = context
    let module = context
//        .compile(program, None)
        .compile(program, None)
//        .expect("Could not compile test program to MLIR.");
        .expect("Could not compile test program to MLIR.");
//

//    assert!(
    assert!(
//        module.module().as_operation().verify(),
        module.module().as_operation().verify(),
//        "Test program generated invalid MLIR:\n{}",
        "Test program generated invalid MLIR:\n{}",
//        module.module().as_operation()
        module.module().as_operation()
//    );
    );
//

//    // FIXME: There are some bugs with non-zero LLVM optimization levels.
    // FIXME: There are some bugs with non-zero LLVM optimization levels.
//    let executor = JitNativeExecutor::from_native_module(module, OptLevel::None);
    let executor = JitNativeExecutor::from_native_module(module, OptLevel::None);
//    match syscall_handler {
    match syscall_handler {
//        Some(syscall_handler) => executor
        Some(syscall_handler) => executor
//            .invoke_dynamic_with_syscall_handler(entry_point_id, args, gas, syscall_handler)
            .invoke_dynamic_with_syscall_handler(entry_point_id, args, gas, syscall_handler)
//            .unwrap(),
            .unwrap(),
//        None => executor.invoke_dynamic(entry_point_id, args, gas).unwrap(),
        None => executor.invoke_dynamic(entry_point_id, args, gas).unwrap(),
//    }
    }
//}
}
//

///// Runs the program on the cairo-vm
/// Runs the program on the cairo-vm
//pub fn run_vm_program(
pub fn run_vm_program(
//    program: &(String, Program, SierraCasmRunner),
    program: &(String, Program, SierraCasmRunner),
//    entry_point: &str,
    entry_point: &str,
//    args: &[Arg],
    args: &[Arg],
//    gas: Option<usize>,
    gas: Option<usize>,
//) -> Result<RunResultStarknet, RunnerError> {
) -> Result<RunResultStarknet, RunnerError> {
//    let runner = &program.2;
    let runner = &program.2;
//    runner.run_function_with_starknet_context(
    runner.run_function_with_starknet_context(
//        runner.find_function(entry_point).unwrap(),
        runner.find_function(entry_point).unwrap(),
//        args,
        args,
//        gas,
        gas,
//        StarknetState::default(),
        StarknetState::default(),
//    )
    )
//}
}
//

//#[track_caller]
#[track_caller]
//pub fn compare_inputless_program(program_path: &str) {
pub fn compare_inputless_program(program_path: &str) {
//    let program: (String, Program, SierraCasmRunner) = load_cairo_path(program_path);
    let program: (String, Program, SierraCasmRunner) = load_cairo_path(program_path);
//    let program = &program;
    let program = &program;
//

//    let result_vm = run_vm_program(program, "main", &[], Some(DEFAULT_GAS as usize)).unwrap();
    let result_vm = run_vm_program(program, "main", &[], Some(DEFAULT_GAS as usize)).unwrap();
//    let result_native = run_native_program(
    let result_native = run_native_program(
//        program,
        program,
//        "main",
        "main",
//        &[],
        &[],
//        Some(DEFAULT_GAS as u128),
        Some(DEFAULT_GAS as u128),
//        Option::<DummySyscallHandler>::None,
        Option::<DummySyscallHandler>::None,
//    );
    );
//

//    compare_outputs(
    compare_outputs(
//        &program.1,
        &program.1,
//        &program.2.find_function("main").unwrap().id,
        &program.2.find_function("main").unwrap().id,
//        &result_vm,
        &result_vm,
//        &result_native,
        &result_native,
//    )
    )
//    .expect("compare error with optlevel none");
    .expect("compare error with optlevel none");
//

//    let result_native = run_native_program(
    let result_native = run_native_program(
//        program,
        program,
//        "main",
        "main",
//        &[],
        &[],
//        Some(DEFAULT_GAS as u128),
        Some(DEFAULT_GAS as u128),
//        Option::<DummySyscallHandler>::None,
        Option::<DummySyscallHandler>::None,
//    );
    );
//

//    compare_outputs(
    compare_outputs(
//        &program.1,
        &program.1,
//        &program.2.find_function("main").unwrap().id,
        &program.2.find_function("main").unwrap().id,
//        &result_vm,
        &result_vm,
//        &result_native,
        &result_native,
//    )
    )
//    .expect("compare error");
    .expect("compare error");
//}
}
//

///// Runs the program using cairo-native JIT.
/// Runs the program using cairo-native JIT.
//pub fn run_native_starknet_contract(
pub fn run_native_starknet_contract(
//    sierra_program: &Program,
    sierra_program: &Program,
//    entry_point_function_idx: usize,
    entry_point_function_idx: usize,
//    args: &[Felt],
    args: &[Felt],
//    handler: impl StarknetSyscallHandler,
    handler: impl StarknetSyscallHandler,
//) -> ContractExecutionResult {
) -> ContractExecutionResult {
//    let native_context = NativeContext::new();
    let native_context = NativeContext::new();
//

//    let native_program = native_context.compile(sierra_program, None).unwrap();
    let native_program = native_context.compile(sierra_program, None).unwrap();
//

//    let entry_point_fn = find_entry_point_by_idx(sierra_program, entry_point_function_idx).unwrap();
    let entry_point_fn = find_entry_point_by_idx(sierra_program, entry_point_function_idx).unwrap();
//    let entry_point_id = &entry_point_fn.id;
    let entry_point_id = &entry_point_fn.id;
//

//    let native_executor = JitNativeExecutor::from_native_module(native_program, Default::default());
    let native_executor = JitNativeExecutor::from_native_module(native_program, Default::default());
//    native_executor
    native_executor
//        .invoke_contract_dynamic(entry_point_id, args, u128::MAX.into(), handler)
        .invoke_contract_dynamic(entry_point_id, args, u128::MAX.into(), handler)
//        .expect("failed to execute the given contract")
        .expect("failed to execute the given contract")
//}
}
//

///// Given the result of the cairo-vm and cairo-native of the same program, it compares
/// Given the result of the cairo-vm and cairo-native of the same program, it compares
///// the results automatically, triggering a proptest assert if there is a mismatch.
/// the results automatically, triggering a proptest assert if there is a mismatch.
/////
///
///// Left of report of the assert is the cairo vm result, right side is cairo native
/// Left of report of the assert is the cairo vm result, right side is cairo native
//#[track_caller]
#[track_caller]
//pub fn compare_outputs(
pub fn compare_outputs(
//    program: &Program,
    program: &Program,
//    entry_point: &FunctionId,
    entry_point: &FunctionId,
//    vm_result: &RunResultStarknet,
    vm_result: &RunResultStarknet,
//    native_result: &ExecutionResult,
    native_result: &ExecutionResult,
//) -> Result<(), TestCaseError> {
) -> Result<(), TestCaseError> {
//    let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(program).unwrap();
    let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(program).unwrap();
//    let function = registry.get_function(entry_point).unwrap();
    let function = registry.get_function(entry_point).unwrap();
//

//    fn map_vm_sizes(
    fn map_vm_sizes(
//        size_cache: &mut HashMap<ConcreteTypeId, usize>,
        size_cache: &mut HashMap<ConcreteTypeId, usize>,
//        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//        ty: &ConcreteTypeId,
        ty: &ConcreteTypeId,
//    ) -> usize {
    ) -> usize {
//        match size_cache.get(ty) {
        match size_cache.get(ty) {
//            Some(&type_size) => type_size,
            Some(&type_size) => type_size,
//            None => {
            None => {
//                let type_size = match registry.get_type(ty).unwrap() {
                let type_size = match registry.get_type(ty).unwrap() {
//                    CoreTypeConcrete::Array(_info) => 2,
                    CoreTypeConcrete::Array(_info) => 2,
//                    CoreTypeConcrete::Felt252(_)
                    CoreTypeConcrete::Felt252(_)
//                    | CoreTypeConcrete::Uint128(_)
                    | CoreTypeConcrete::Uint128(_)
//                    | CoreTypeConcrete::Uint64(_)
                    | CoreTypeConcrete::Uint64(_)
//                    | CoreTypeConcrete::Uint32(_)
                    | CoreTypeConcrete::Uint32(_)
//                    | CoreTypeConcrete::Uint16(_)
                    | CoreTypeConcrete::Uint16(_)
//                    | CoreTypeConcrete::Uint8(_)
                    | CoreTypeConcrete::Uint8(_)
//                    | CoreTypeConcrete::Sint128(_)
                    | CoreTypeConcrete::Sint128(_)
//                    | CoreTypeConcrete::Sint64(_)
                    | CoreTypeConcrete::Sint64(_)
//                    | CoreTypeConcrete::Sint32(_)
                    | CoreTypeConcrete::Sint32(_)
//                    | CoreTypeConcrete::Sint16(_)
                    | CoreTypeConcrete::Sint16(_)
//                    | CoreTypeConcrete::Sint8(_) => 1,
                    | CoreTypeConcrete::Sint8(_) => 1,
//                    CoreTypeConcrete::Enum(info) => {
                    CoreTypeConcrete::Enum(info) => {
//                        1 + info
                        1 + info
//                            .variants
                            .variants
//                            .iter()
                            .iter()
//                            .map(|variant_ty| map_vm_sizes(size_cache, registry, variant_ty))
                            .map(|variant_ty| map_vm_sizes(size_cache, registry, variant_ty))
//                            .max()
                            .max()
//                            .unwrap_or_default()
                            .unwrap_or_default()
//                    }
                    }
//                    CoreTypeConcrete::Struct(info) => info
                    CoreTypeConcrete::Struct(info) => info
//                        .members
                        .members
//                        .iter()
                        .iter()
//                        .map(|member_ty| map_vm_sizes(size_cache, registry, member_ty))
                        .map(|member_ty| map_vm_sizes(size_cache, registry, member_ty))
//                        .sum(),
                        .sum(),
//                    CoreTypeConcrete::Nullable(_) => 1,
                    CoreTypeConcrete::Nullable(_) => 1,
//                    CoreTypeConcrete::NonZero(info) => map_vm_sizes(size_cache, registry, &info.ty),
                    CoreTypeConcrete::NonZero(info) => map_vm_sizes(size_cache, registry, &info.ty),
//                    CoreTypeConcrete::EcPoint(_) => 2,
                    CoreTypeConcrete::EcPoint(_) => 2,
//                    CoreTypeConcrete::EcState(_) => 4,
                    CoreTypeConcrete::EcState(_) => 4,
//                    CoreTypeConcrete::Snapshot(info) => {
                    CoreTypeConcrete::Snapshot(info) => {
//                        map_vm_sizes(size_cache, registry, &info.ty)
                        map_vm_sizes(size_cache, registry, &info.ty)
//                    }
                    }
//                    x => todo!("vm size not yet implemented: {:?}", x.info()),
                    x => todo!("vm size not yet implemented: {:?}", x.info()),
//                };
                };
//                size_cache.insert(ty.clone(), type_size);
                size_cache.insert(ty.clone(), type_size);
//

//                type_size
                type_size
//            }
            }
//        }
        }
//    }
    }
//

//    fn map_vm_values(
    fn map_vm_values(
//        size_cache: &mut HashMap<ConcreteTypeId, usize>,
        size_cache: &mut HashMap<ConcreteTypeId, usize>,
//        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//        memory: &[Option<Felt252>],
        memory: &[Option<Felt252>],
//        mut values: &[Felt252],
        mut values: &[Felt252],
//        ty: &ConcreteTypeId,
        ty: &ConcreteTypeId,
//    ) -> JitValue {
    ) -> JitValue {
//        match registry.get_type(ty).unwrap() {
        match registry.get_type(ty).unwrap() {
//            CoreTypeConcrete::Array(info) => {
            CoreTypeConcrete::Array(info) => {
//                assert_eq!(values.len(), 2);
                assert_eq!(values.len(), 2);
//                let since_ptr = values[0].to_usize().unwrap();
                let since_ptr = values[0].to_usize().unwrap();
//                let until_ptr = values[1].to_usize().unwrap();
                let until_ptr = values[1].to_usize().unwrap();
//

//                let total_len = until_ptr - since_ptr;
                let total_len = until_ptr - since_ptr;
//                let elem_size = map_vm_sizes(size_cache, registry, &info.ty);
                let elem_size = map_vm_sizes(size_cache, registry, &info.ty);
//                assert_eq!(total_len % elem_size, 0);
                assert_eq!(total_len % elem_size, 0);
//

//                JitValue::Array(
                JitValue::Array(
//                    memory[since_ptr..until_ptr]
                    memory[since_ptr..until_ptr]
//                        .chunks(elem_size)
                        .chunks(elem_size)
//                        .map(|data| data.iter().cloned().map(Option::unwrap).collect::<Vec<_>>())
                        .map(|data| data.iter().cloned().map(Option::unwrap).collect::<Vec<_>>())
//                        .map(|data| map_vm_values(size_cache, registry, memory, &data, &info.ty))
                        .map(|data| map_vm_values(size_cache, registry, memory, &data, &info.ty))
//                        .collect(),
                        .collect(),
//                )
                )
//            }
            }
//            CoreTypeConcrete::Felt252(_) => {
            CoreTypeConcrete::Felt252(_) => {
//                JitValue::Felt252(Felt::from_bytes_le(&values[0].to_le_bytes()))
                JitValue::Felt252(Felt::from_bytes_le(&values[0].to_le_bytes()))
//            }
            }
//            CoreTypeConcrete::Uint128(_) => JitValue::Uint128(values[0].to_u128().unwrap()),
            CoreTypeConcrete::Uint128(_) => JitValue::Uint128(values[0].to_u128().unwrap()),
//            CoreTypeConcrete::Uint64(_) => JitValue::Uint64(values[0].to_u64().unwrap()),
            CoreTypeConcrete::Uint64(_) => JitValue::Uint64(values[0].to_u64().unwrap()),
//            CoreTypeConcrete::Uint32(_) => JitValue::Uint32(values[0].to_u32().unwrap()),
            CoreTypeConcrete::Uint32(_) => JitValue::Uint32(values[0].to_u32().unwrap()),
//            CoreTypeConcrete::Uint16(_) => JitValue::Uint16(values[0].to_u16().unwrap()),
            CoreTypeConcrete::Uint16(_) => JitValue::Uint16(values[0].to_u16().unwrap()),
//            CoreTypeConcrete::Uint8(_) => JitValue::Uint8(values[0].to_u8().unwrap()),
            CoreTypeConcrete::Uint8(_) => JitValue::Uint8(values[0].to_u8().unwrap()),
//            CoreTypeConcrete::Sint128(_) => {
            CoreTypeConcrete::Sint128(_) => {
//                JitValue::Sint128(if values[0].to_bigint() >= *HALF_PRIME {
                JitValue::Sint128(if values[0].to_bigint() >= *HALF_PRIME {
//                    -(&*PRIME - &values[0].to_biguint()).to_i128().unwrap()
                    -(&*PRIME - &values[0].to_biguint()).to_i128().unwrap()
//                } else {
                } else {
//                    values[0].to_biguint().to_i128().unwrap()
                    values[0].to_biguint().to_i128().unwrap()
//                })
                })
//            }
            }
//            CoreTypeConcrete::Sint64(_) => {
            CoreTypeConcrete::Sint64(_) => {
//                JitValue::Sint64(if values[0].to_bigint() >= *HALF_PRIME {
                JitValue::Sint64(if values[0].to_bigint() >= *HALF_PRIME {
//                    -(&*PRIME - &values[0].to_biguint()).to_i64().unwrap()
                    -(&*PRIME - &values[0].to_biguint()).to_i64().unwrap()
//                } else {
                } else {
//                    values[0].to_biguint().to_i64().unwrap()
                    values[0].to_biguint().to_i64().unwrap()
//                })
                })
//            }
            }
//            CoreTypeConcrete::Sint32(_) => {
            CoreTypeConcrete::Sint32(_) => {
//                JitValue::Sint32(if values[0].to_bigint() >= *HALF_PRIME {
                JitValue::Sint32(if values[0].to_bigint() >= *HALF_PRIME {
//                    -(&*PRIME - &values[0].to_biguint()).to_i32().unwrap()
                    -(&*PRIME - &values[0].to_biguint()).to_i32().unwrap()
//                } else {
                } else {
//                    values[0].to_biguint().to_i32().unwrap()
                    values[0].to_biguint().to_i32().unwrap()
//                })
                })
//            }
            }
//            CoreTypeConcrete::Sint16(_) => {
            CoreTypeConcrete::Sint16(_) => {
//                JitValue::Sint16(if values[0].to_bigint() >= *HALF_PRIME {
                JitValue::Sint16(if values[0].to_bigint() >= *HALF_PRIME {
//                    -(&*PRIME - &values[0].to_biguint()).to_i16().unwrap()
                    -(&*PRIME - &values[0].to_biguint()).to_i16().unwrap()
//                } else {
                } else {
//                    values[0].to_biguint().to_i16().unwrap()
                    values[0].to_biguint().to_i16().unwrap()
//                })
                })
//            }
            }
//            CoreTypeConcrete::Sint8(_) => {
            CoreTypeConcrete::Sint8(_) => {
//                JitValue::Sint8(if values[0].to_bigint() >= *HALF_PRIME {
                JitValue::Sint8(if values[0].to_bigint() >= *HALF_PRIME {
//                    -(&*PRIME - &values[0].to_biguint()).to_i8().unwrap()
                    -(&*PRIME - &values[0].to_biguint()).to_i8().unwrap()
//                } else {
                } else {
//                    values[0].to_biguint().to_i8().unwrap()
                    values[0].to_biguint().to_i8().unwrap()
//                })
                })
//            }
            }
//            CoreTypeConcrete::Enum(info) => {
            CoreTypeConcrete::Enum(info) => {
//                let enum_size = map_vm_sizes(size_cache, registry, ty);
                let enum_size = map_vm_sizes(size_cache, registry, ty);
//                assert_eq!(values.len(), enum_size);
                assert_eq!(values.len(), enum_size);
//

//                let (tag, data);
                let (tag, data);
//                (tag, values) = values.split_first().unwrap();
                (tag, values) = values.split_first().unwrap();
//

//                let mut tag = tag.to_usize().unwrap();
                let mut tag = tag.to_usize().unwrap();
//                if info.variants.len() > 2 {
                if info.variants.len() > 2 {
//                    tag = info.variants.len() - ((tag + 1) >> 1);
                    tag = info.variants.len() - ((tag + 1) >> 1);
//                }
                }
//                assert!(tag <= info.variants.len());
                assert!(tag <= info.variants.len());
//                data = &values[enum_size - size_cache[&info.variants[tag]] - 1..];
                data = &values[enum_size - size_cache[&info.variants[tag]] - 1..];
//

//                JitValue::Enum {
                JitValue::Enum {
//                    tag,
                    tag,
//                    value: Box::new(map_vm_values(
                    value: Box::new(map_vm_values(
//                        size_cache,
                        size_cache,
//                        registry,
                        registry,
//                        memory,
                        memory,
//                        data,
                        data,
//                        &info.variants[tag],
                        &info.variants[tag],
//                    )),
                    )),
//                    debug_name: ty.debug_name.as_deref().map(String::from),
                    debug_name: ty.debug_name.as_deref().map(String::from),
//                }
                }
//            }
            }
//            CoreTypeConcrete::Struct(info) => JitValue::Struct {
            CoreTypeConcrete::Struct(info) => JitValue::Struct {
//                fields: info
                fields: info
//                    .members
                    .members
//                    .iter()
                    .iter()
//                    .map(|member_ty| {
                    .map(|member_ty| {
//                        let data;
                        let data;
//                        (data, values) =
                        (data, values) =
//                            values.split_at(map_vm_sizes(size_cache, registry, member_ty));
                            values.split_at(map_vm_sizes(size_cache, registry, member_ty));
//

//                        map_vm_values(size_cache, registry, memory, data, member_ty)
                        map_vm_values(size_cache, registry, memory, data, member_ty)
//                    })
                    })
//                    .collect(),
                    .collect(),
//                debug_name: ty.debug_name.as_deref().map(String::from),
                debug_name: ty.debug_name.as_deref().map(String::from),
//            },
            },
//            CoreTypeConcrete::SquashedFelt252Dict(info) => JitValue::Felt252Dict {
            CoreTypeConcrete::SquashedFelt252Dict(info) => JitValue::Felt252Dict {
//                value: (values[0].to_usize().unwrap()..values[1].to_usize().unwrap())
                value: (values[0].to_usize().unwrap()..values[1].to_usize().unwrap())
//                    .step_by(3)
                    .step_by(3)
//                    .map(|index| {
                    .map(|index| {
//                        (
                        (
//                            Felt::from_bytes_le(&memory[index].clone().unwrap().to_le_bytes()),
                            Felt::from_bytes_le(&memory[index].clone().unwrap().to_le_bytes()),
//                            match &info.info.long_id.generic_args[0] {
                            match &info.info.long_id.generic_args[0] {
//                                cairo_lang_sierra::program::GenericArg::Type(ty) => map_vm_values(
                                cairo_lang_sierra::program::GenericArg::Type(ty) => map_vm_values(
//                                    size_cache,
                                    size_cache,
//                                    registry,
                                    registry,
//                                    memory,
                                    memory,
//                                    &[memory[index + 2].clone().unwrap()],
                                    &[memory[index + 2].clone().unwrap()],
//                                    ty,
                                    ty,
//                                ),
                                ),
//                                _ => unimplemented!("unsupported dict value type"),
                                _ => unimplemented!("unsupported dict value type"),
//                            },
                            },
//                        )
                        )
//                    })
                    })
//                    .collect(),
                    .collect(),
//                debug_name: ty.debug_name.as_deref().map(String::from),
                debug_name: ty.debug_name.as_deref().map(String::from),
//            },
            },
//            CoreTypeConcrete::Snapshot(info) => {
            CoreTypeConcrete::Snapshot(info) => {
//                map_vm_values(size_cache, registry, memory, values, &info.ty)
                map_vm_values(size_cache, registry, memory, values, &info.ty)
//            }
            }
//            CoreTypeConcrete::Nullable(info) => {
            CoreTypeConcrete::Nullable(info) => {
//                assert_eq!(values.len(), 1);
                assert_eq!(values.len(), 1);
//

//                let ty_size = map_vm_sizes(size_cache, registry, &info.ty);
                let ty_size = map_vm_sizes(size_cache, registry, &info.ty);
//                match values[0].to_usize().unwrap() {
                match values[0].to_usize().unwrap() {
//                    0 => JitValue::Null,
                    0 => JitValue::Null,
//                    ptr if ty_size == 0 => {
                    ptr if ty_size == 0 => {
//                        assert_eq!(ptr, 1);
                        assert_eq!(ptr, 1);
//                        map_vm_values(size_cache, registry, memory, &[], &info.ty)
                        map_vm_values(size_cache, registry, memory, &[], &info.ty)
//                    }
                    }
//                    ptr => map_vm_values(
                    ptr => map_vm_values(
//                        size_cache,
                        size_cache,
//                        registry,
                        registry,
//                        memory,
                        memory,
//                        &memory[ptr..ptr + ty_size]
                        &memory[ptr..ptr + ty_size]
//                            .iter()
                            .iter()
//                            .cloned()
                            .cloned()
//                            .map(Option::unwrap)
                            .map(Option::unwrap)
//                            .collect::<Vec<_>>(),
                            .collect::<Vec<_>>(),
//                        &info.ty,
                        &info.ty,
//                    ),
                    ),
//                }
                }
//            }
            }
//            CoreTypeConcrete::NonZero(info) => {
            CoreTypeConcrete::NonZero(info) => {
//                map_vm_values(size_cache, registry, memory, values, &info.ty)
                map_vm_values(size_cache, registry, memory, values, &info.ty)
//            }
            }
//            CoreTypeConcrete::EcPoint(_) => {
            CoreTypeConcrete::EcPoint(_) => {
//                assert_eq!(values.len(), 2);
                assert_eq!(values.len(), 2);
//

//                JitValue::EcPoint(
                JitValue::EcPoint(
//                    Felt::from_bytes_le(&values[0].to_le_bytes()),
                    Felt::from_bytes_le(&values[0].to_le_bytes()),
//                    Felt::from_bytes_le(&values[1].to_le_bytes()),
                    Felt::from_bytes_le(&values[1].to_le_bytes()),
//                )
                )
//            }
            }
//            CoreTypeConcrete::EcState(_) => {
            CoreTypeConcrete::EcState(_) => {
//                assert_eq!(values.len(), 4);
                assert_eq!(values.len(), 4);
//

//                JitValue::EcState(
                JitValue::EcState(
//                    Felt::from_bytes_le(&values[0].to_le_bytes()),
                    Felt::from_bytes_le(&values[0].to_le_bytes()),
//                    Felt::from_bytes_le(&values[1].to_le_bytes()),
                    Felt::from_bytes_le(&values[1].to_le_bytes()),
//                    Felt::from_bytes_le(&values[2].to_le_bytes()),
                    Felt::from_bytes_le(&values[2].to_le_bytes()),
//                    Felt::from_bytes_le(&values[3].to_le_bytes()),
                    Felt::from_bytes_le(&values[3].to_le_bytes()),
//                )
                )
//            }
            }
//            CoreTypeConcrete::Bytes31(_) => {
            CoreTypeConcrete::Bytes31(_) => {
//                let mut bytes = values[0].to_le_bytes().to_vec();
                let mut bytes = values[0].to_le_bytes().to_vec();
//                bytes.pop();
                bytes.pop();
//                JitValue::Bytes31(bytes.try_into().unwrap())
                JitValue::Bytes31(bytes.try_into().unwrap())
//            }
            }
//            CoreTypeConcrete::Const(_) => todo!(),
            CoreTypeConcrete::Const(_) => todo!(),
//            CoreTypeConcrete::BoundedInt(_) => todo!(),
            CoreTypeConcrete::BoundedInt(_) => todo!(),
//            CoreTypeConcrete::Coupon(_) => todo!(),
            CoreTypeConcrete::Coupon(_) => todo!(),
//            x => {
            x => {
//                todo!("vm value not yet implemented: {:?}", x.info())
                todo!("vm value not yet implemented: {:?}", x.info())
//            }
            }
//        }
        }
//    }
    }
//

//    let mut size_cache = HashMap::new();
    let mut size_cache = HashMap::new();
//    let ty = function.signature.ret_types.last();
    let ty = function.signature.ret_types.last();
//    let returns_panic = ty.map_or(false, |ty| {
    let returns_panic = ty.map_or(false, |ty| {
//        ty.debug_name
        ty.debug_name
//            .as_ref()
            .as_ref()
//            .map(|x| x.starts_with("core::panics::PanicResult"))
            .map(|x| x.starts_with("core::panics::PanicResult"))
//            .unwrap_or(false)
            .unwrap_or(false)
//    });
    });
//    assert_eq!(
    assert_eq!(
//        vm_result
        vm_result
//            .gas_counter
            .gas_counter
//            .clone()
            .clone()
//            .unwrap_or_else(|| Felt252::from(0)),
            .unwrap_or_else(|| Felt252::from(0)),
//        Felt252::from(native_result.remaining_gas.unwrap_or(0)),
        Felt252::from(native_result.remaining_gas.unwrap_or(0)),
//    );
    );
//

//    let vm_result = match &vm_result.value {
    let vm_result = match &vm_result.value {
//        RunResultValue::Success(values) if !values.is_empty() | returns_panic => {
        RunResultValue::Success(values) if !values.is_empty() | returns_panic => {
//            if returns_panic {
            if returns_panic {
//                let inner_ty = match registry.get_type(ty.unwrap())? {
                let inner_ty = match registry.get_type(ty.unwrap())? {
//                    CoreTypeConcrete::Enum(info) => &info.variants[0],
                    CoreTypeConcrete::Enum(info) => &info.variants[0],
//                    _ => unreachable!(),
                    _ => unreachable!(),
//                };
                };
//                JitValue::Enum {
                JitValue::Enum {
//                    tag: 0,
                    tag: 0,
//                    value: Box::new(map_vm_values(
                    value: Box::new(map_vm_values(
//                        &mut size_cache,
                        &mut size_cache,
//                        &registry,
                        &registry,
//                        &vm_result.memory,
                        &vm_result.memory,
//                        values,
                        values,
//                        inner_ty,
                        inner_ty,
//                    )),
                    )),
//                    debug_name: None,
                    debug_name: None,
//                }
                }
//            } else {
            } else {
//                map_vm_values(
                map_vm_values(
//                    &mut size_cache,
                    &mut size_cache,
//                    &registry,
                    &registry,
//                    &vm_result.memory,
                    &vm_result.memory,
//                    values,
                    values,
//                    ty.unwrap(),
                    ty.unwrap(),
//                )
                )
//            }
            }
//        }
        }
//        RunResultValue::Panic(values) => JitValue::Enum {
        RunResultValue::Panic(values) => JitValue::Enum {
//            tag: 1,
            tag: 1,
//            value: Box::new(JitValue::Struct {
            value: Box::new(JitValue::Struct {
//                fields: vec![
                fields: vec![
//                    JitValue::Struct {
                    JitValue::Struct {
//                        fields: Vec::new(),
                        fields: Vec::new(),
//                        debug_name: None,
                        debug_name: None,
//                    },
                    },
//                    JitValue::Array(
                    JitValue::Array(
//                        values
                        values
//                            .iter()
                            .iter()
//                            .map(|value| Felt::from_bytes_le(&value.to_le_bytes()))
                            .map(|value| Felt::from_bytes_le(&value.to_le_bytes()))
//                            .map(JitValue::Felt252)
                            .map(JitValue::Felt252)
//                            .collect(),
                            .collect(),
//                    ),
                    ),
//                ],
                ],
//                debug_name: None,
                debug_name: None,
//            }),
            }),
//            debug_name: None,
            debug_name: None,
//        },
        },
//        // Empty return value
        // Empty return value
//        _ => JitValue::Struct {
        _ => JitValue::Struct {
//            fields: vec![],
            fields: vec![],
//            debug_name: None,
            debug_name: None,
//        },
        },
//    };
    };
//

//    pretty_assertions_sorted::assert_eq!(native_result.return_value, vm_result);
    pretty_assertions_sorted::assert_eq!(native_result.return_value, vm_result);
//    Ok(())
    Ok(())
//}
}
//

//pub const FIELD_HIGH: u128 = (1 << 123) + (17 << 64); // this is equal to 10633823966279327296825105735305134080
pub const FIELD_HIGH: u128 = (1 << 123) + (17 << 64); // this is equal to 10633823966279327296825105735305134080
//pub const FIELD_LOW: u128 = 1;
pub const FIELD_LOW: u128 = 1;
//

///// Returns a [`Strategy`] that generates any valid Felt
/// Returns a [`Strategy`] that generates any valid Felt
//pub fn any_felt() -> impl Strategy<Value = Felt> {
pub fn any_felt() -> impl Strategy<Value = Felt> {
//    use proptest::prelude::*;
    use proptest::prelude::*;
//

//    (0..=FIELD_HIGH)
    (0..=FIELD_HIGH)
//        // turn range into `impl Strategy`
        // turn range into `impl Strategy`
//        .prop_map(|x| x)
        .prop_map(|x| x)
//        // choose second 128-bit limb capped by first one
        // choose second 128-bit limb capped by first one
//        .prop_flat_map(|high| {
        .prop_flat_map(|high| {
//            let low = if high == FIELD_HIGH {
            let low = if high == FIELD_HIGH {
//                (0..FIELD_LOW).prop_map(|x| x).sboxed()
                (0..FIELD_LOW).prop_map(|x| x).sboxed()
//            } else {
            } else {
//                any::<u128>().sboxed()
                any::<u128>().sboxed()
//            };
            };
//            (Just(high), low)
            (Just(high), low)
//        })
        })
//        // turn (u128, u128) into limbs array and then into Felt
        // turn (u128, u128) into limbs array and then into Felt
//        .prop_map(|(high, low)| {
        .prop_map(|(high, low)| {
//            let limbs = [
            let limbs = [
//                (high >> 64) as u64,
                (high >> 64) as u64,
//                (high & ((1 << 64) - 1)) as u64,
                (high & ((1 << 64) - 1)) as u64,
//                (low >> 64) as u64,
                (low >> 64) as u64,
//                (low & ((1 << 64) - 1)) as u64,
                (low & ((1 << 64) - 1)) as u64,
//            ];
            ];
//            FieldElement::new(UnsignedInteger::from_limbs(limbs))
            FieldElement::new(UnsignedInteger::from_limbs(limbs))
//        })
        })
//        .prop_map(|value: FieldElement<MontgomeryBackendPrimeField<_, 4>>| {
        .prop_map(|value: FieldElement<MontgomeryBackendPrimeField<_, 4>>| {
//            Felt::from_bytes_be(&value.to_bytes_be())
            Felt::from_bytes_be(&value.to_bytes_be())
//        })
        })
//}
}
//

///// Returns a [`Strategy`] that generates any nonzero Felt
/// Returns a [`Strategy`] that generates any nonzero Felt
//pub fn nonzero_felt() -> impl Strategy<Value = Felt> {
pub fn nonzero_felt() -> impl Strategy<Value = Felt> {
//    any_felt().prop_filter("is zero", |x| x != &Felt::ZERO)
    any_felt().prop_filter("is zero", |x| x != &Felt::ZERO)
//}
}
