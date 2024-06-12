//use crate::common::{
use crate::common::{
//    any_felt, compare_outputs, load_cairo, run_native_program, run_vm_program, DEFAULT_GAS,
    any_felt, compare_outputs, load_cairo, run_native_program, run_vm_program, DEFAULT_GAS,
//};
};
//use cairo_felt::Felt252 as DeprecatedFelt;
use cairo_felt::Felt252 as DeprecatedFelt;
//use cairo_lang_runner::{Arg, SierraCasmRunner};
use cairo_lang_runner::{Arg, SierraCasmRunner};
//use cairo_lang_sierra::program::Program;
use cairo_lang_sierra::program::Program;
//use cairo_native::{starknet::DummySyscallHandler, values::JitValue};
use cairo_native::{starknet::DummySyscallHandler, values::JitValue};
//use lazy_static::lazy_static;
use lazy_static::lazy_static;
//use num_bigint::BigUint;
use num_bigint::BigUint;
//use proptest::prelude::*;
use proptest::prelude::*;
//use starknet_types_core::felt::Felt;
use starknet_types_core::felt::Felt;
//use std::str::FromStr;
use std::str::FromStr;
//

//lazy_static! {
lazy_static! {
//    static ref EC_POINT_TRY_NEW: (String, Program, SierraCasmRunner) = load_cairo! {
    static ref EC_POINT_TRY_NEW: (String, Program, SierraCasmRunner) = load_cairo! {
//        use core::{ec::{ec_point_try_new_nz, EcPoint}};
        use core::{ec::{ec_point_try_new_nz, EcPoint}};
//        use core::zeroable::NonZero;
        use core::zeroable::NonZero;
//

//        fn run_test(x: felt252, y: felt252) -> Option<NonZero<EcPoint>> {
        fn run_test(x: felt252, y: felt252) -> Option<NonZero<EcPoint>> {
//            ec_point_try_new_nz(x, y)
            ec_point_try_new_nz(x, y)
//        }
        }
//    };
    };
//    static ref EC_POINT_FROM_X: (String, Program, SierraCasmRunner) = load_cairo! {
    static ref EC_POINT_FROM_X: (String, Program, SierraCasmRunner) = load_cairo! {
//        use core::{ec::{ec_point_from_x_nz, EcPoint}};
        use core::{ec::{ec_point_from_x_nz, EcPoint}};
//        use core::zeroable::NonZero;
        use core::zeroable::NonZero;
//

//        fn run_test(x: felt252) -> Option<NonZero<EcPoint>> {
        fn run_test(x: felt252) -> Option<NonZero<EcPoint>> {
//            ec_point_from_x_nz(x)
            ec_point_from_x_nz(x)
//        }
        }
//    };
    };
//    static ref EC_POINT_ZERO: (String, Program, SierraCasmRunner) = load_cairo! {
    static ref EC_POINT_ZERO: (String, Program, SierraCasmRunner) = load_cairo! {
//        use core::ec::{ec_point_zero, EcPoint};
        use core::ec::{ec_point_zero, EcPoint};
//

//        fn run_test() -> EcPoint {
        fn run_test() -> EcPoint {
//            ec_point_zero()
            ec_point_zero()
//        }
        }
//    };
    };
//}
}
//

//#[test]
#[test]
//fn ec_point_zero() {
fn ec_point_zero() {
//    let program = &EC_POINT_ZERO;
    let program = &EC_POINT_ZERO;
//    let result_vm = run_vm_program(program, "run_test", &[], Some(DEFAULT_GAS as usize)).unwrap();
    let result_vm = run_vm_program(program, "run_test", &[], Some(DEFAULT_GAS as usize)).unwrap();
//    let result_native = run_native_program(
    let result_native = run_native_program(
//        program,
        program,
//        "run_test",
        "run_test",
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
//        &program.2.find_function("run_test").unwrap().id,
        &program.2.find_function("run_test").unwrap().id,
//        &result_vm,
        &result_vm,
//        &result_native,
        &result_native,
//    )
    )
//    .unwrap();
    .unwrap();
//}
}
//

//#[test]
#[test]
//fn ec_point_from_x_big() {
fn ec_point_from_x_big() {
//    let x = DeprecatedFelt::new(
    let x = DeprecatedFelt::new(
//        BigUint::from_str(
        BigUint::from_str(
//            "10503791839462130483045092717244804953879649418761481950933471772092536173",
            "10503791839462130483045092717244804953879649418761481950933471772092536173",
//        )
        )
//        .unwrap(),
        .unwrap(),
//    );
    );
//    let program = &EC_POINT_FROM_X;
    let program = &EC_POINT_FROM_X;
//    let result_vm = run_vm_program(
    let result_vm = run_vm_program(
//        program,
        program,
//        "run_test",
        "run_test",
//        &[Arg::Value(x.clone())],
        &[Arg::Value(x.clone())],
//        Some(DEFAULT_GAS as usize),
        Some(DEFAULT_GAS as usize),
//    )
    )
//    .unwrap();
    .unwrap();
//    let result_native = run_native_program(
    let result_native = run_native_program(
//        program,
        program,
//        "run_test",
        "run_test",
//        &[JitValue::Felt252(Felt::from_bytes_be(&x.to_be_bytes()))],
        &[JitValue::Felt252(Felt::from_bytes_be(&x.to_be_bytes()))],
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
//        &program.2.find_function("run_test").unwrap().id,
        &program.2.find_function("run_test").unwrap().id,
//        &result_vm,
        &result_vm,
//        &result_native,
        &result_native,
//    )
    )
//    .unwrap();
    .unwrap();
//}
}
//

//#[test]
#[test]
//fn ec_point_from_x_small() {
fn ec_point_from_x_small() {
//    let x = DeprecatedFelt::new(BigUint::from_str("1234").unwrap());
    let x = DeprecatedFelt::new(BigUint::from_str("1234").unwrap());
//    let program = &EC_POINT_FROM_X;
    let program = &EC_POINT_FROM_X;
//    let result_vm = run_vm_program(
    let result_vm = run_vm_program(
//        program,
        program,
//        "run_test",
        "run_test",
//        &[Arg::Value(x.clone())],
        &[Arg::Value(x.clone())],
//        Some(DEFAULT_GAS as usize),
        Some(DEFAULT_GAS as usize),
//    )
    )
//    .unwrap();
    .unwrap();
//    let result_native = run_native_program(
    let result_native = run_native_program(
//        program,
        program,
//        "run_test",
        "run_test",
//        &[JitValue::Felt252(Felt::from_bytes_be(&x.to_be_bytes()))],
        &[JitValue::Felt252(Felt::from_bytes_be(&x.to_be_bytes()))],
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
//        &program.2.find_function("run_test").unwrap().id,
        &program.2.find_function("run_test").unwrap().id,
//        &result_vm,
        &result_vm,
//        &result_native,
        &result_native,
//    )
    )
//    .unwrap();
    .unwrap();
//}
}
//

//proptest! {
proptest! {
//    #[test]
    #[test]
//    fn ec_point_try_new_proptest(a in any_felt(), b in any_felt()) {
    fn ec_point_try_new_proptest(a in any_felt(), b in any_felt()) {
//        let program = &EC_POINT_TRY_NEW;
        let program = &EC_POINT_TRY_NEW;
//        let result_vm = run_vm_program(
        let result_vm = run_vm_program(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[Arg::Value(DeprecatedFelt::from_bytes_be(&a.clone().to_bytes_be())), Arg::Value(DeprecatedFelt::from_bytes_be(&b.clone().to_bytes_be()))],
            &[Arg::Value(DeprecatedFelt::from_bytes_be(&a.clone().to_bytes_be())), Arg::Value(DeprecatedFelt::from_bytes_be(&b.clone().to_bytes_be()))],
//            Some(DEFAULT_GAS as usize),
            Some(DEFAULT_GAS as usize),
//        )
        )
//        .unwrap();
        .unwrap();
//        let result_native = run_native_program(
        let result_native = run_native_program(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[JitValue::Felt252(a), JitValue::Felt252(b)],
            &[JitValue::Felt252(a), JitValue::Felt252(b)],
//            Some(DEFAULT_GAS as u128),
            Some(DEFAULT_GAS as u128),
//            Option::<DummySyscallHandler>::None,
            Option::<DummySyscallHandler>::None,
//        );
        );
//

//        compare_outputs(
        compare_outputs(
//            &program.1,
            &program.1,
//            &program.2.find_function("run_test").unwrap().id,
            &program.2.find_function("run_test").unwrap().id,
//            &result_vm,
            &result_vm,
//            &result_native,
            &result_native,
//        )?;
        )?;
//    }
    }
//

//    #[test]
    #[test]
//    fn ec_point_from_x_proptest(a in any_felt()) {
    fn ec_point_from_x_proptest(a in any_felt()) {
//        let program = &EC_POINT_FROM_X;
        let program = &EC_POINT_FROM_X;
//        let result_vm = run_vm_program(
        let result_vm = run_vm_program(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[Arg::Value(DeprecatedFelt::from_bytes_be(&a.clone().to_bytes_be()))],
            &[Arg::Value(DeprecatedFelt::from_bytes_be(&a.clone().to_bytes_be()))],
//            Some(DEFAULT_GAS as usize),
            Some(DEFAULT_GAS as usize),
//        )
        )
//        .unwrap();
        .unwrap();
//        let result_native = run_native_program(
        let result_native = run_native_program(
//            program,
            program,
//            "run_test",
            "run_test",
//            &[JitValue::Felt252(a)],
            &[JitValue::Felt252(a)],
//            Some(DEFAULT_GAS as u128),
            Some(DEFAULT_GAS as u128),
//            Option::<DummySyscallHandler>::None,
            Option::<DummySyscallHandler>::None,
//        );
        );
//

//        compare_outputs(
        compare_outputs(
//            &program.1,
            &program.1,
//            &program.2.find_function("run_test").unwrap().id,
            &program.2.find_function("run_test").unwrap().id,
//            &result_vm,
            &result_vm,
//            &result_native,
            &result_native,
//        )?;
        )?;
//    }
    }
//}
}
