//use crate::common::{
use crate::common::{
//    any_felt, compare_outputs, load_cairo, nonzero_felt, run_native_program, run_vm_program,
    any_felt, compare_outputs, load_cairo, nonzero_felt, run_native_program, run_vm_program,
//    DEFAULT_GAS,
    DEFAULT_GAS,
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
//use proptest::prelude::*;
use proptest::prelude::*;
//

//lazy_static! {
lazy_static! {
//

//    static ref FELT252_ADD: (String, Program, SierraCasmRunner) = load_cairo! {
    static ref FELT252_ADD: (String, Program, SierraCasmRunner) = load_cairo! {
//        fn run_test(lhs: felt252, rhs: felt252) -> felt252 {
        fn run_test(lhs: felt252, rhs: felt252) -> felt252 {
//            lhs + rhs
            lhs + rhs
//        }
        }
//    };
    };
//

//    static ref FELT252_SUB: (String, Program, SierraCasmRunner) = load_cairo! {
    static ref FELT252_SUB: (String, Program, SierraCasmRunner) = load_cairo! {
//        fn run_test(lhs: felt252, rhs: felt252) -> felt252 {
        fn run_test(lhs: felt252, rhs: felt252) -> felt252 {
//            lhs - rhs
            lhs - rhs
//        }
        }
//    };
    };
//

//    static ref FELT252_MUL: (String, Program, SierraCasmRunner) = load_cairo! {
    static ref FELT252_MUL: (String, Program, SierraCasmRunner) = load_cairo! {
//        fn run_test(lhs: felt252, rhs: felt252) -> felt252 {
        fn run_test(lhs: felt252, rhs: felt252) -> felt252 {
//            lhs * rhs
            lhs * rhs
//        }
        }
//    };
    };
//

//    static ref FELT252_DIV: (String, Program, SierraCasmRunner) = load_cairo! {
    static ref FELT252_DIV: (String, Program, SierraCasmRunner) = load_cairo! {
//        fn run_test(lhs: felt252, rhs: felt252) -> felt252 {
        fn run_test(lhs: felt252, rhs: felt252) -> felt252 {
//            felt252_div(lhs, rhs.try_into().unwrap())
            felt252_div(lhs, rhs.try_into().unwrap())
//        }
        }
//    };
    };
//

//    // TODO: Add test program for `felt252_add_const`.
    // TODO: Add test program for `felt252_add_const`.
//    // TODO: Add test program for `felt252_sub_const`.
    // TODO: Add test program for `felt252_sub_const`.
//    // TODO: Add test program for `felt252_mul_const`.
    // TODO: Add test program for `felt252_mul_const`.
//    // TODO: Add test program for `felt252_div_const`.
    // TODO: Add test program for `felt252_div_const`.
//

//    static ref FELT252_CONST: (String, Program, SierraCasmRunner) = load_cairo! {
    static ref FELT252_CONST: (String, Program, SierraCasmRunner) = load_cairo! {
//        fn run_test() -> (felt252, felt252, felt252, felt252) {
        fn run_test() -> (felt252, felt252, felt252, felt252) {
//            (0, 1, -2, -1)
            (0, 1, -2, -1)
//        }
        }
//    };
    };
//

//    static ref FELT252_IS_ZERO: (String, Program, SierraCasmRunner) = load_cairo! {
    static ref FELT252_IS_ZERO: (String, Program, SierraCasmRunner) = load_cairo! {
//        fn run_test(x: felt252) -> felt252 {
        fn run_test(x: felt252) -> felt252 {
//            match x {
            match x {
//                0 => 1,
                0 => 1,
//                _ => 0,
                _ => 0,
//            }
            }
//        }
        }
//    };
    };
//}
}
//

//proptest! {
proptest! {
//    #[test]
    #[test]
//    fn felt_add_proptest(a in any_felt(), b in any_felt()) {
    fn felt_add_proptest(a in any_felt(), b in any_felt()) {
//        let program = &FELT252_ADD;
        let program = &FELT252_ADD;
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
//    fn felt_sub_proptest(a in any_felt(), b in any_felt()) {
    fn felt_sub_proptest(a in any_felt(), b in any_felt()) {
//        let program = &FELT252_SUB;
        let program = &FELT252_SUB;
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
//    fn felt_mul_proptest(a in any_felt(), b in any_felt()) {
    fn felt_mul_proptest(a in any_felt(), b in any_felt()) {
//        let program = &FELT252_MUL;
        let program = &FELT252_MUL;
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
//    fn felt_div_proptest(a in any_felt(), b in nonzero_felt()) {
    fn felt_div_proptest(a in any_felt(), b in nonzero_felt()) {
//        let program = &FELT252_DIV;
        let program = &FELT252_DIV;
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
//}
}
