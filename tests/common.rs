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
    extensions::core::{CoreLibfunc, CoreType, CoreTypeConcrete},
    ids::FunctionId,
    program::{GenericArg, Program},
    program_registry::ProgramRegistry,
};
use cairo_lang_sierra_generator::replace_ids::DebugReplacer;
use cairo_lang_starknet::contract::get_contracts_info;
use cairo_native::{
    context::NativeContext,
    execution_result::ContractExecutionResult,
    executor::NativeExecutor,
    metadata::{
        gas::{GasMetadata, MetadataComputationConfig},
        runtime_bindings::RuntimeBindingsMeta,
        syscall_handler::SyscallHandlerMeta,
        MetadataStorage,
    },
    starknet::StarkNetSyscallHandler,
    types::felt252::PRIME,
    utils::{find_entry_point_by_idx, register_runtime_symbols, run_pass_manager},
    values::JITValue,
    ExecutionResult,
};
use lambdaworks_math::{
    field::{
        element::FieldElement, fields::montgomery_backed_prime_fields::MontgomeryBackendPrimeField,
    },
    unsigned_integer::element::UnsignedInteger,
};
use melior::{
    dialect::DialectRegistry,
    ir::{Location, Module},
    utility::{register_all_dialects, register_all_passes},
    Context, ExecutionEngine,
};
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::identities::Zero;
use proptest::{strategy::Strategy, test_runner::TestCaseError};
use starknet_types_core::felt::{felt_to_biguint, Felt};
use std::{
    env::var, fs, iter::Peekable, ops::Neg, path::Path, slice::Iter, str::FromStr, sync::Arc,
};

#[allow(unused_macros)]
macro_rules! load_cairo {
    ( $( $program:tt )+ ) => {
        $crate::common::load_cairo_str(stringify!($($program)+))
    };
}

#[allow(unused_imports)]
pub(crate) use load_cairo;

pub(crate) const GAS: u64 = u64::MAX;

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
        RunResultValue::Success(x) => x.iter().map(|x| x.to_string()).collect::<Vec<_>>(),
        RunResultValue::Panic(x) => x.iter().map(|x| x.to_string()).collect::<Vec<_>>(),
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
    let program = Arc::try_unwrap(
        compile_prepared_db(
            &mut db,
            main_crate_ids.clone(),
            CompilerConfig {
                replace_ids: true,
                ..Default::default()
            },
        )
        .unwrap(),
    )
    .unwrap();

    let module_name = program_file.path().with_extension("");
    let module_name = module_name.file_name().unwrap().to_str().unwrap();

    let replacer = DebugReplacer { db: &db };
    let contracts_info = get_contracts_info(&db, main_crate_ids, &replacer).unwrap();

    let runner =
        SierraCasmRunner::new(program.clone(), Some(Default::default()), contracts_info).unwrap();

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
    let program = Arc::try_unwrap(
        compile_prepared_db(
            &mut db,
            main_crate_ids.clone(),
            CompilerConfig {
                replace_ids: true,
                ..Default::default()
            },
        )
        .unwrap(),
    )
    .unwrap();

    let module_name = program_file.with_extension("");
    let module_name = module_name.file_name().unwrap().to_str().unwrap();

    let replacer = DebugReplacer { db: &db };
    let contracts_info = get_contracts_info(&db, main_crate_ids, &replacer).unwrap();

    let runner =
        SierraCasmRunner::new(program.clone(), Some(Default::default()), contracts_info).unwrap();

    (module_name.to_string(), program, runner)
}

/// Runs the program using cairo-native JIT.
pub fn run_native_program(
    program: &(String, Program, SierraCasmRunner),
    entry_point: &str,
    args: &[JITValue],
) -> ExecutionResult {
    let entry_point = format!("{0}::{0}::{1}", program.0, entry_point);
    let program = &program.1;

    let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(program)
        .expect("Could not create the test program registry.");

    let entry_point_id = &program
        .funcs
        .iter()
        .find(|x| x.id.debug_name.as_deref() == Some(&entry_point))
        .expect("Test program entry point not found.")
        .id;

    let context = Context::new();
    context.append_dialect_registry(&{
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        registry
    });
    context.load_all_available_dialects();
    register_all_passes();

    let mut module = Module::new(Location::unknown(&context));

    let mut metadata = MetadataStorage::new();
    // Make the runtime library available.
    metadata.insert(RuntimeBindingsMeta::default()).unwrap();

    // Gas
    let required_initial_gas = if program
        .type_declarations
        .iter()
        .any(|decl| decl.long_id.generic_id.0.as_str() == "GasBuiltin")
    {
        let gas_metadata = GasMetadata::new(program, MetadataComputationConfig::default());

        let required_initial_gas = { gas_metadata.get_initial_required_gas(entry_point_id) };
        metadata.insert(gas_metadata).unwrap();
        required_initial_gas
    } else {
        None
    };

    cairo_native::compile::<CoreType, CoreLibfunc>(
        &context,
        &module,
        program,
        &registry,
        &mut metadata,
        None,
    )
    .expect("Could not compile test program to MLIR.");

    assert!(
        module.as_operation().verify(),
        "Test program generated invalid MLIR:\n{}",
        module.as_operation()
    );

    run_pass_manager(&context, &mut module)
        .expect("Could not apply passes to the compiled test program.");

    let engine = ExecutionEngine::new(&module, 0, &[], false);

    #[cfg(feature = "with-runtime")]
    register_runtime_symbols(&engine);

    cairo_native::execute(
        &engine,
        &registry,
        &program
            .funcs
            .iter()
            .find(|x| x.id.debug_name.as_deref() == Some(&entry_point))
            .expect("Test program entry point not found.")
            .id,
        args,
        required_initial_gas,
        Some(u64::MAX.into()),
        None,
    )
    .expect("Test program execution failed.")
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

#[track_caller]
pub fn compare_inputless_program(program_path: &str) {
    let program: (String, Program, SierraCasmRunner) = load_cairo_path(program_path);
    let program = &program;

    let result_vm = run_vm_program(program, "main", &[], Some(GAS as usize)).unwrap();

    let result_native = run_native_program(program, "main", &[]);

    compare_outputs(
        &program.1,
        &program.2.find_function("main").unwrap().id,
        &result_vm,
        &result_native,
    )
    .expect("compare error");
}

/// Runs the program using cairo-native JIT.
pub fn run_native_starknet_contract<T>(
    sierra_program: &Program,
    entry_point_function_idx: usize,
    args: &[JITValue],
    handler: &mut T,
) -> ContractExecutionResult
where
    T: StarkNetSyscallHandler,
{
    let native_context = NativeContext::new();

    let mut native_program = native_context.compile(sierra_program).unwrap();
    native_program
        .insert_metadata(SyscallHandlerMeta::new(handler))
        .unwrap();

    let entry_point_fn = find_entry_point_by_idx(sierra_program, entry_point_function_idx).unwrap();
    let entry_point_id = &entry_point_fn.id;

    let native_executor = NativeExecutor::new(native_program);

    native_executor
        .execute_contract(entry_point_id, args, u64::MAX as u128)
        .expect("failed to execute the given contract")
}

/// Given the result of the cairo-vm and cairo-native of the same program, it compares
/// the results automatically, triggering a proptest assert if there is a mismatch.
///
/// If ignore_gas is false, it will check whether the resulting gas matches.
///
/// Left of report of the assert is the cairo vm result, right side is cairo native
#[track_caller]
pub fn compare_outputs(
    program: &Program,
    entry_point: &FunctionId,
    vm_result: &RunResultStarknet,
    native_result: &ExecutionResult,
) -> Result<(), TestCaseError> {
    use proptest::prelude::*;

    let reg: ProgramRegistry<CoreType, CoreLibfunc> =
        ProgramRegistry::new(program).expect("program registry should be created");

    let func = reg
        .get_function(entry_point)
        .expect("entry point should exist");

    let ret_types = &func.signature.ret_types;
    let mut native_rets = native_result.return_values.iter().peekable();
    let vm_return_vals = get_run_result(&vm_result.value);
    let mut vm_rets = vm_return_vals.iter().peekable();
    let vm_gas: u128 = vm_result
        .gas_counter
        .as_ref()
        .map(|x| x.to_biguint().try_into().unwrap())
        .unwrap_or(0);
    let native_gas = native_result.remaining_gas.unwrap_or(0);
    prop_assert_eq!(vm_gas, native_gas, "gas mismatch");

    #[track_caller]
    fn check_next_type<'a>(
        ty: &CoreTypeConcrete,
        native_rets: &mut impl Iterator<Item = &'a JITValue>,
        vm_rets: &mut Peekable<Iter<'_, String>>,
        reg: &ProgramRegistry<CoreType, CoreLibfunc>,
    ) -> Result<(), TestCaseError> {
        let mut native_rets = native_rets.into_iter().peekable();
        match ty {
            CoreTypeConcrete::Array(info) => {
                if let JITValue::Array(data) = native_rets.next().unwrap() {
                    for value in data {
                        check_next_type(
                            reg.get_type(&info.ty).expect("type should exist"),
                            &mut [value].into_iter(),
                            vm_rets,
                            reg,
                        )?;
                    }
                } else {
                    panic!("invalid type")
                }
            }
            CoreTypeConcrete::Bitwise(_) => {
                // ignore
            }
            CoreTypeConcrete::Box(_) => todo!(),
            CoreTypeConcrete::EcOp(_) => todo!(),
            CoreTypeConcrete::EcPoint(_) => {
                let native_data = native_rets.next().unwrap();

                if let JITValue::EcPoint(a, b) = native_data {
                    let vm_a = BigUint::from_str(vm_rets.next().unwrap()).unwrap();
                    let vm_b = BigUint::from_str(vm_rets.next().unwrap()).unwrap();

                    let native_value_a = felt_to_biguint(*a);
                    let native_value_b = felt_to_biguint(*b);
                    prop_assert_eq!(vm_a, native_value_a, "felt value mismatch in ecpoint");
                    prop_assert_eq!(vm_b, native_value_b, "felt value mismatch in ecpoint");
                } else {
                    panic!("invalid type")
                }
            }
            CoreTypeConcrete::EcState(_) => todo!(),
            CoreTypeConcrete::Felt252(_) => {
                prop_assert!(vm_rets.peek().is_some(), "cairo-vm missing next value");
                prop_assert!(
                    native_rets.peek().is_some(),
                    "cairo-native missing next value"
                );
                let vm_value = vm_rets.next().unwrap();

                if let JITValue::Felt252(felt) = native_rets.next().unwrap() {
                    let native_value = felt_to_biguint(*felt);
                    let vm_value = BigUint::from_str(vm_value).unwrap();
                    prop_assert_eq!(vm_value, native_value, "felt value mismatch");
                } else {
                    prop_assert!(false, "invalid felt value type");
                }
            }
            CoreTypeConcrete::GasBuiltin(_) => {
                // ignore
            }
            CoreTypeConcrete::BuiltinCosts(_) => todo!(),
            CoreTypeConcrete::Uint8(_) => {
                prop_assert!(vm_rets.peek().is_some(), "cairo-vm missing next value");
                prop_assert!(
                    native_rets.peek().is_some(),
                    "cairo-native missing next value"
                );
                let vm_value: u8 = vm_rets.next().unwrap().parse().unwrap();
                let native_value: u8 = if let JITValue::Uint8(v) = native_rets.next().unwrap() {
                    *v
                } else {
                    panic!("invalid type")
                };
                prop_assert_eq!(vm_value, native_value)
            }
            CoreTypeConcrete::Uint16(_) => {
                prop_assert!(vm_rets.peek().is_some(), "cairo-vm missing next value");
                prop_assert!(
                    native_rets.peek().is_some(),
                    "cairo-native missing next value"
                );
                let vm_value: u16 = vm_rets.next().unwrap().parse().unwrap();
                let native_value: u16 = if let JITValue::Uint16(v) = native_rets.next().unwrap() {
                    *v
                } else {
                    panic!("invalid type")
                };
                prop_assert_eq!(vm_value, native_value)
            }
            CoreTypeConcrete::Uint32(_) => {
                prop_assert!(vm_rets.peek().is_some(), "cairo-vm missing next value");
                prop_assert!(
                    native_rets.peek().is_some(),
                    "cairo-native missing next value"
                );
                let vm_value: u32 = vm_rets.next().unwrap().parse().unwrap();
                let native_value: u32 = if let JITValue::Uint32(v) = native_rets.next().unwrap() {
                    *v
                } else {
                    panic!("invalid type")
                };
                prop_assert_eq!(vm_value, native_value)
            }
            CoreTypeConcrete::Uint64(_) => {
                prop_assert!(vm_rets.peek().is_some(), "cairo-vm missing next value");
                prop_assert!(
                    native_rets.peek().is_some(),
                    "cairo-native missing next value"
                );
                let vm_value: u64 = vm_rets.next().unwrap().parse().unwrap();
                let native_value: u64 = if let JITValue::Uint64(v) = native_rets.next().unwrap() {
                    *v
                } else {
                    panic!("invalid type")
                };
                prop_assert_eq!(vm_value, native_value)
            }
            CoreTypeConcrete::Uint128(_) => {
                prop_assert!(vm_rets.peek().is_some(), "cairo-vm missing next value");
                prop_assert!(
                    native_rets.peek().is_some(),
                    "cairo-native missing next value"
                );
                let vm_value: u128 = vm_rets.next().unwrap().parse().unwrap();
                let native_value: u128 = if let JITValue::Uint128(v) = native_rets.next().unwrap() {
                    *v
                } else {
                    panic!("invalid type")
                };
                prop_assert_eq!(vm_value, native_value)
            }
            CoreTypeConcrete::Uint128MulGuarantee(_) => todo!(),
            CoreTypeConcrete::NonZero(_) => todo!(),
            CoreTypeConcrete::Nullable(_) => todo!(),
            CoreTypeConcrete::RangeCheck(_) => {
                // runner: ignore
                // native: null
            }
            CoreTypeConcrete::Uninitialized(_) => todo!(),
            CoreTypeConcrete::Enum(info) => {
                // cairo-vm fills with 0 before the variant value based on the variant with most values, if they are a tuple
                /*
                   enum MyEnum {
                       A: Felt,
                       B: (Felt, Felt, Felt),
                   }

                   will produce [0, 0, A_value] for the variant A in cairo-vm, because B is a tuple of 3 elements.
                   0 is a filler.
                */
                prop_assert!(native_rets.peek().is_some());
                if let JITValue::Enum {
                    tag: native_tag,
                    value,
                    ..
                } = native_rets.next().unwrap()
                {
                    let native_tag = *native_tag;
                    let mut is_bool = false;

                    if let GenericArg::UserType(id) = &info.info.long_id.generic_args[0] {
                        is_bool = id.debug_name.as_ref().unwrap().as_str().eq("core::bool");
                    }

                    let is_panic = match &info.info.long_id.generic_args[0] {
                        GenericArg::UserType(info) => info
                            .debug_name
                            .as_ref()
                            .unwrap()
                            .as_str()
                            .starts_with("core::panics::PanicResult"),
                        _ => false,
                    };

                    if is_bool {
                        let vn_val = vm_rets.next().unwrap() == "1";
                        let native_val: bool = native_tag == 1; // 1 = true
                        prop_assert_eq!(vn_val, native_val, "bool value mismatch");
                    } else if is_panic {
                        check_next_type(
                            reg.get_type(&info.variants[native_tag]).unwrap(),
                            &mut [&**value].into_iter(),
                            vm_rets,
                            reg,
                        )?;
                    } else {
                        let vm_tag = {
                            let vm_tag = vm_rets.next().unwrap();
                            if info.variants.len() > 2 {
                                casm_variant_to_sierra(
                                    vm_tag.parse::<i64>().unwrap(),
                                    info.variants.len() as i64,
                                ) as u64
                            } else {
                                vm_tag.parse().unwrap()
                            }
                        };
                        prop_assert_eq!(vm_tag, native_tag as u64, "non panic enum tag mismatch");

                        let mut max_tuple_count = 0;
                        let mut this_variant_count = 0;

                        // check if a variant is a struct, we need to skip cairo-vm values if the current
                        // variant has less elements count than the biggest struct field count.
                        for (i, x) in info.variants.iter().enumerate() {
                            let variant_type = reg.get_type(x).unwrap();

                            let current_count = if let CoreTypeConcrete::Struct(struct_type) =
                                variant_type
                            {
                                max_tuple_count = max_tuple_count.max(struct_type.members.len());
                                struct_type.members.len()
                            } else {
                                max_tuple_count = max_tuple_count.max(1);
                                1
                            };

                            if i == native_tag {
                                this_variant_count = current_count;
                            }
                        }

                        // can't use skip because changes the iterator type and it's lazy.
                        for _ in 0..(max_tuple_count - this_variant_count) {
                            vm_rets.next().expect("should have filler value");
                        }

                        check_next_type(
                            reg.get_type(&info.variants[native_tag]).unwrap(),
                            &mut [&**value].into_iter(),
                            vm_rets,
                            reg,
                        )?;
                    }
                } else {
                    panic!("wrong type, should be a enum")
                };
            }
            CoreTypeConcrete::Struct(info) => {
                if let JITValue::Struct { fields, .. } = native_rets.next().unwrap() {
                    for (field, container) in info.members.iter().zip(fields.iter()) {
                        check_next_type(
                            reg.get_type(field).unwrap(),
                            &mut [container].into_iter(),
                            vm_rets,
                            reg,
                        )?;
                    }
                } else {
                    panic!("wrong type, should be a struct")
                }
            }
            CoreTypeConcrete::Felt252Dict(_) => todo!(),
            CoreTypeConcrete::Felt252DictEntry(_) => todo!(),
            CoreTypeConcrete::SquashedFelt252Dict(_) => todo!(),
            CoreTypeConcrete::Pedersen(_) => {
                // ignore
            }
            CoreTypeConcrete::Poseidon(_) => {
                // ignore
            }
            CoreTypeConcrete::Span(_) => todo!(),
            CoreTypeConcrete::StarkNet(_) => todo!(),
            CoreTypeConcrete::SegmentArena(_) => {
                // ignore
            }
            CoreTypeConcrete::Snapshot(_) => todo!(),
            CoreTypeConcrete::Sint8(_) => todo!(),
            CoreTypeConcrete::Sint16(_) => todo!(),
            CoreTypeConcrete::Sint32(_) => todo!(),
            CoreTypeConcrete::Sint64(_) => todo!(),
            CoreTypeConcrete::Sint128(_) => todo!(),
            CoreTypeConcrete::Bytes31(_) => todo!(),
        }

        Ok(())
    }

    for ty in ret_types {
        let ty = reg.get_type(ty).unwrap();
        check_next_type(ty, &mut native_rets, &mut vm_rets, &reg)?;
    }

    Ok(())
}

pub const FIELD_HIGH: u128 = (1 << 123) + (17 << 64); // this is equal to 10633823966279327296825105735305134080
pub const FIELD_LOW: u128 = 1;

/// Returns a [`Strategy`] that generates any valid Felt
pub fn any_felt252() -> impl Strategy<Value = Felt> {
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
pub fn nonzero_felt252() -> impl Strategy<Value = Felt> {
    any_felt252().prop_filter("is zero", |x| !x.is_zero())
}
