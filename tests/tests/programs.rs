//use crate::common::{any_felt, load_cairo, run_native_program, run_vm_program};
use crate::common::{any_felt, load_cairo, run_native_program, run_vm_program};
//use crate::common::{compare_outputs, DEFAULT_GAS};
use crate::common::{compare_outputs, DEFAULT_GAS};
//use cairo_felt::Felt252 as DeprecatedFelt;
use cairo_felt::Felt252 as DeprecatedFelt;
//use cairo_lang_runner::{Arg, SierraCasmRunner};
use cairo_lang_runner::{Arg, SierraCasmRunner};
//use cairo_lang_sierra::program::Program;
use cairo_lang_sierra::program::Program;
//use cairo_native::starknet::DummySyscallHandler;
use cairo_native::starknet::DummySyscallHandler;
//use cairo_native::values::JitValue;
use cairo_native::values::JitValue;
//use lazy_static::lazy_static;
use lazy_static::lazy_static;
//use num_traits::Num;
use num_traits::Num;
//use proptest::prelude::*;
use proptest::prelude::*;
//

//lazy_static! {
lazy_static! {
//    pub static ref FACTORIAL: (String, Program, SierraCasmRunner) = load_cairo! {
    pub static ref FACTORIAL: (String, Program, SierraCasmRunner) = load_cairo! {
//        fn factorial(value: felt252, n: felt252) -> felt252 {
        fn factorial(value: felt252, n: felt252) -> felt252 {
//            if (n == 1) {
            if (n == 1) {
//                value
                value
//            } else {
            } else {
//                factorial(value * n, n - 1)
                factorial(value * n, n - 1)
//            }
            }
//        }
        }
//

//        fn run_test(n: felt252) -> felt252 {
        fn run_test(n: felt252) -> felt252 {
//            factorial(1, n)
            factorial(1, n)
//        }
        }
//    };
    };
//

//    pub static ref FIB: (String, Program, SierraCasmRunner) = load_cairo! {
    pub static ref FIB: (String, Program, SierraCasmRunner) = load_cairo! {
//        fn fib(a: felt252, b: felt252, n: felt252) -> felt252 {
        fn fib(a: felt252, b: felt252, n: felt252) -> felt252 {
//            match n {
            match n {
//                0 => a,
                0 => a,
//                _ => fib(b, a + b, n - 1),
                _ => fib(b, a + b, n - 1),
//            }
            }
//        }
        }
//

//        fn run_test(n: felt252) -> felt252 {
        fn run_test(n: felt252) -> felt252 {
//            fib(0, 1, n)
            fib(0, 1, n)
//        }
        }
//    };
    };
//

//    pub static ref LOGISTIC_MAP: (String, Program, SierraCasmRunner) = load_cairo! {
    pub static ref LOGISTIC_MAP: (String, Program, SierraCasmRunner) = load_cairo! {
//        fn iterate_map(r: felt252, x: felt252) -> felt252 {
        fn iterate_map(r: felt252, x: felt252) -> felt252 {
//            r * x * -x
            r * x * -x
//        }
        }
//

//        // good default: 1000
        // good default: 1000
//        fn run_test(mut i: felt252) -> felt252 {
        fn run_test(mut i: felt252) -> felt252 {
//            // Initial value.
            // Initial value.
//            let mut x = 1234567890123456789012345678901234567890;
            let mut x = 1234567890123456789012345678901234567890;
//

//            // Iterate the map.
            // Iterate the map.
//            loop {
            loop {
//                x = iterate_map(4, x);
                x = iterate_map(4, x);
//

//                if i == 0 {
                if i == 0 {
//                    break x;
                    break x;
//                }
                }
//

//                i = i - 1;
                i = i - 1;
//            }
            }
//        }
        }
//    };
    };
//

//    pub static ref PEDERSEN: (String, Program, SierraCasmRunner) = load_cairo! {
    pub static ref PEDERSEN: (String, Program, SierraCasmRunner) = load_cairo! {
//        use core::pedersen::pedersen;
        use core::pedersen::pedersen;
//

//        fn run_test(a: felt252, b: felt252) -> felt252 {
        fn run_test(a: felt252, b: felt252) -> felt252 {
//            pedersen(a, b)
            pedersen(a, b)
//        }
        }
//    };
    };
//

//    pub static ref POSEIDON: (String, Program, SierraCasmRunner) = load_cairo! {
    pub static ref POSEIDON: (String, Program, SierraCasmRunner) = load_cairo! {
//        use core::poseidon::hades_permutation;
        use core::poseidon::hades_permutation;
//

//        fn run_test(a: felt252, b: felt252, c: felt252) -> (felt252, felt252, felt252) {
        fn run_test(a: felt252, b: felt252, c: felt252) -> (felt252, felt252, felt252) {
//            hades_permutation(a, b, c)
            hades_permutation(a, b, c)
//        }
        }
//    };
    };
//

//    pub static ref SELF_REFERENCING: (String, Program, SierraCasmRunner) = load_cairo! {
    pub static ref SELF_REFERENCING: (String, Program, SierraCasmRunner) = load_cairo! {
//        #[derive(Drop, Copy, PartialEq)]
        #[derive(Drop, Copy, PartialEq)]
//        enum ArrayItem {
        enum ArrayItem {
//            Span: Span<u8>,
            Span: Span<u8>,
//            Recursive: Span<ArrayItem>
            Recursive: Span<ArrayItem>
//        }
        }
//

//        fn recursion(input: Span<u8>) -> Span<ArrayItem> {
        fn recursion(input: Span<u8>) -> Span<ArrayItem> {
//            let mut output: Array<ArrayItem> = Default::default();
            let mut output: Array<ArrayItem> = Default::default();
//

//            let index = (*input.at(0));
            let index = (*input.at(0));
//            if index < 5 {
            if index < 5 {
//                output.append(ArrayItem::Span(input));
                output.append(ArrayItem::Span(input));
//            } else {
            } else {
//                let res = recursion(input.slice(1, input.len() - 1));
                let res = recursion(input.slice(1, input.len() - 1));
//                output.append(ArrayItem::Recursive(res));
                output.append(ArrayItem::Recursive(res));
//            }
            }
//

//            return output.span();
            return output.span();
//        }
        }
//

//        fn run_test() -> Span<ArrayItem> {
        fn run_test() -> Span<ArrayItem> {
//            let arr = array![10, 9, 8, 7, 6, 4];
            let arr = array![10, 9, 8, 7, 6, 4];
//            recursion(arr.span())
            recursion(arr.span())
//        }
        }
//    };
    };
//

//    pub static ref NO_OP: (String, Program, SierraCasmRunner) = load_cairo! {
    pub static ref NO_OP: (String, Program, SierraCasmRunner) = load_cairo! {
//        #[inline(never)]
        #[inline(never)]
//        fn no_op() {}
        fn no_op() {}
//

//        fn run_test() {
        fn run_test() {
//            no_op();
            no_op();
//        }
        }
//    };
    };
//}
}
//

//#[test]
#[test]
//fn fib() {
fn fib() {
//    let result_vm = run_vm_program(
    let result_vm = run_vm_program(
//        &FIB,
        &FIB,
//        "run_test",
        "run_test",
//        &[Arg::Value(DeprecatedFelt::from(10))],
        &[Arg::Value(DeprecatedFelt::from(10))],
//        Some(DEFAULT_GAS as usize),
        Some(DEFAULT_GAS as usize),
//    )
    )
//    .unwrap();
    .unwrap();
//    let result_native = run_native_program(
    let result_native = run_native_program(
//        &FIB,
        &FIB,
//        "run_test",
        "run_test",
//        &[JitValue::Felt252(10.into())],
        &[JitValue::Felt252(10.into())],
//        Some(DEFAULT_GAS as u128),
        Some(DEFAULT_GAS as u128),
//        Option::<DummySyscallHandler>::None,
        Option::<DummySyscallHandler>::None,
//    );
    );
//

//    compare_outputs(
    compare_outputs(
//        &FIB.1,
        &FIB.1,
//        &FIB.2.find_function("run_test").unwrap().id,
        &FIB.2.find_function("run_test").unwrap().id,
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
//fn logistic_map() {
fn logistic_map() {
//    let result_vm = run_vm_program(
    let result_vm = run_vm_program(
//        &LOGISTIC_MAP,
        &LOGISTIC_MAP,
//        "run_test",
        "run_test",
//        &[Arg::Value(DeprecatedFelt::from(1000))],
        &[Arg::Value(DeprecatedFelt::from(1000))],
//        Some(DEFAULT_GAS as usize),
        Some(DEFAULT_GAS as usize),
//    )
    )
//    .unwrap();
    .unwrap();
//    let result_native = run_native_program(
    let result_native = run_native_program(
//        &LOGISTIC_MAP,
        &LOGISTIC_MAP,
//        "run_test",
        "run_test",
//        &[JitValue::Felt252(1000.into())],
        &[JitValue::Felt252(1000.into())],
//        Some(DEFAULT_GAS as u128),
        Some(DEFAULT_GAS as u128),
//        Option::<DummySyscallHandler>::None,
        Option::<DummySyscallHandler>::None,
//    );
    );
//

//    compare_outputs(
    compare_outputs(
//        &LOGISTIC_MAP.1,
        &LOGISTIC_MAP.1,
//        &LOGISTIC_MAP.2.find_function("run_test").unwrap().id,
        &LOGISTIC_MAP.2.find_function("run_test").unwrap().id,
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
//fn pedersen() {
fn pedersen() {
//    let result_vm = run_vm_program(
    let result_vm = run_vm_program(
//        &PEDERSEN,
        &PEDERSEN,
//        "run_test",
        "run_test",
//        &[
        &[
//            Arg::Value(
            Arg::Value(
//                DeprecatedFelt::from_str_radix(
                DeprecatedFelt::from_str_radix(
//                    "2163739901324492107409690946633517860331020929182861814098856895601180685",
                    "2163739901324492107409690946633517860331020929182861814098856895601180685",
//                    10,
                    10,
//                )
                )
//                .unwrap(),
                .unwrap(),
//            ),
            ),
//            Arg::Value(
            Arg::Value(
//                DeprecatedFelt::from_str_radix(
                DeprecatedFelt::from_str_radix(
//                    "2392090257937917229310563411601744459500735555884672871108624696010915493156",
                    "2392090257937917229310563411601744459500735555884672871108624696010915493156",
//                    10,
                    10,
//                )
                )
//                .unwrap(),
                .unwrap(),
//            ),
            ),
//        ],
        ],
//        Some(DEFAULT_GAS as usize),
        Some(DEFAULT_GAS as usize),
//    )
    )
//    .unwrap();
    .unwrap();
//    let result_native = run_native_program(
    let result_native = run_native_program(
//        &PEDERSEN,
        &PEDERSEN,
//        "run_test",
        "run_test",
//        &[
        &[
//            JitValue::felt_str(
            JitValue::felt_str(
//                "2163739901324492107409690946633517860331020929182861814098856895601180685",
                "2163739901324492107409690946633517860331020929182861814098856895601180685",
//            ),
            ),
//            JitValue::felt_str(
            JitValue::felt_str(
//                "2392090257937917229310563411601744459500735555884672871108624696010915493156",
                "2392090257937917229310563411601744459500735555884672871108624696010915493156",
//            ),
            ),
//        ],
        ],
//        Some(DEFAULT_GAS as u128),
        Some(DEFAULT_GAS as u128),
//        Option::<DummySyscallHandler>::None,
        Option::<DummySyscallHandler>::None,
//    );
    );
//

//    compare_outputs(
    compare_outputs(
//        &PEDERSEN.1,
        &PEDERSEN.1,
//        &PEDERSEN.2.find_function("run_test").unwrap().id,
        &PEDERSEN.2.find_function("run_test").unwrap().id,
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
//fn factorial() {
fn factorial() {
//    let result_vm = run_vm_program(
    let result_vm = run_vm_program(
//        &FACTORIAL,
        &FACTORIAL,
//        "run_test",
        "run_test",
//        &[Arg::Value(DeprecatedFelt::from(13))],
        &[Arg::Value(DeprecatedFelt::from(13))],
//        Some(DEFAULT_GAS as usize),
        Some(DEFAULT_GAS as usize),
//    )
    )
//    .unwrap();
    .unwrap();
//    let result_native = run_native_program(
    let result_native = run_native_program(
//        &FACTORIAL,
        &FACTORIAL,
//        "run_test",
        "run_test",
//        &[JitValue::Felt252(13.into())],
        &[JitValue::Felt252(13.into())],
//        Some(DEFAULT_GAS as u128),
        Some(DEFAULT_GAS as u128),
//        Option::<DummySyscallHandler>::None,
        Option::<DummySyscallHandler>::None,
//    );
    );
//

//    compare_outputs(
    compare_outputs(
//        &FACTORIAL.1,
        &FACTORIAL.1,
//        &FACTORIAL.2.find_function("run_test").unwrap().id,
        &FACTORIAL.2.find_function("run_test").unwrap().id,
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
//    fn fib_proptest(n in 0..100i32) {
    fn fib_proptest(n in 0..100i32) {
//        let result_vm = run_vm_program(
        let result_vm = run_vm_program(
//            &FIB,
            &FIB,
//            "run_test",
            "run_test",
//            &[Arg::Value(DeprecatedFelt::from(n))],
            &[Arg::Value(DeprecatedFelt::from(n))],
//            Some(DEFAULT_GAS as usize),
            Some(DEFAULT_GAS as usize),
//        )
        )
//        .unwrap();
        .unwrap();
//        let result_native = run_native_program(
        let result_native = run_native_program(
//            &FIB,
            &FIB,
//            "run_test",
            "run_test",
//            &[JitValue::Felt252(n.into())],
            &[JitValue::Felt252(n.into())],
//            Some(DEFAULT_GAS as u128),
            Some(DEFAULT_GAS as u128),
//            Option::<DummySyscallHandler>::None,
            Option::<DummySyscallHandler>::None,
//        );
        );
//

//        compare_outputs(
        compare_outputs(
//            &FIB.1,
            &FIB.1,
//            &FIB.2.find_function("run_test").unwrap().id,
            &FIB.2.find_function("run_test").unwrap().id,
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
//    fn logistic_map_proptest(n in 100..110i32) {
    fn logistic_map_proptest(n in 100..110i32) {
//        let result_vm = run_vm_program(
        let result_vm = run_vm_program(
//            &LOGISTIC_MAP,
            &LOGISTIC_MAP,
//            "run_test",
            "run_test",
//            &[Arg::Value(DeprecatedFelt::from(n))],
            &[Arg::Value(DeprecatedFelt::from(n))],
//            Some(DEFAULT_GAS as usize),
            Some(DEFAULT_GAS as usize),
//        )
        )
//        .unwrap();
        .unwrap();
//        let result_native = run_native_program(
        let result_native = run_native_program(
//            &LOGISTIC_MAP,
            &LOGISTIC_MAP,
//            "run_test",
            "run_test",
//            &[JitValue::Felt252(n.into())],
            &[JitValue::Felt252(n.into())],
//            Some(DEFAULT_GAS as u128),
            Some(DEFAULT_GAS as u128),
//            Option::<DummySyscallHandler>::None,
            Option::<DummySyscallHandler>::None,
//        );
        );
//

//        compare_outputs(
        compare_outputs(
//            &LOGISTIC_MAP.1,
            &LOGISTIC_MAP.1,
//            &LOGISTIC_MAP.2.find_function("run_test").unwrap().id,
            &LOGISTIC_MAP.2.find_function("run_test").unwrap().id,
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
//    fn factorial_proptest(n in 1..100i32) {
    fn factorial_proptest(n in 1..100i32) {
//        let result_vm = run_vm_program(
        let result_vm = run_vm_program(
//            &FACTORIAL,
            &FACTORIAL,
//            "run_test",
            "run_test",
//            &[Arg::Value(DeprecatedFelt::from(n))],
            &[Arg::Value(DeprecatedFelt::from(n))],
//            Some(DEFAULT_GAS as usize),
            Some(DEFAULT_GAS as usize),
//        )
        )
//        .unwrap();
        .unwrap();
//        let result_native = run_native_program(
        let result_native = run_native_program(
//            &FACTORIAL,
            &FACTORIAL,
//            "run_test",
            "run_test",
//            &[JitValue::Felt252(n.into())],
            &[JitValue::Felt252(n.into())],
//            Some(DEFAULT_GAS as u128),
            Some(DEFAULT_GAS as u128),
//            Option::<DummySyscallHandler>::None,
            Option::<DummySyscallHandler>::None,
//        );
        );
//

//        compare_outputs(
        compare_outputs(
//            &FACTORIAL.1,
            &FACTORIAL.1,
//            &FACTORIAL.2.find_function("run_test").unwrap().id,
            &FACTORIAL.2.find_function("run_test").unwrap().id,
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
//    fn pedersen_proptest(a in any_felt(), b in any_felt()) {
    fn pedersen_proptest(a in any_felt(), b in any_felt()) {
//        let result_vm = run_vm_program(
        let result_vm = run_vm_program(
//            &PEDERSEN,
            &PEDERSEN,
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
//

//        let result_native = run_native_program(
        let result_native = run_native_program(
//            &PEDERSEN,
            &PEDERSEN,
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
//            &PEDERSEN.1,
            &PEDERSEN.1,
//            &PEDERSEN.2.find_function("run_test").unwrap().id,
            &PEDERSEN.2.find_function("run_test").unwrap().id,
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
//    fn poseidon_proptest(a in any_felt(), b in any_felt(), c in any_felt()) {
    fn poseidon_proptest(a in any_felt(), b in any_felt(), c in any_felt()) {
//        let result_vm = run_vm_program(
        let result_vm = run_vm_program(
//            &POSEIDON,
            &POSEIDON,
//            "run_test",
            "run_test",
//            &[Arg::Value(DeprecatedFelt::from_bytes_be(&a.clone().to_bytes_be())),
            &[Arg::Value(DeprecatedFelt::from_bytes_be(&a.clone().to_bytes_be())),
//             Arg::Value(DeprecatedFelt::from_bytes_be(&b.clone().to_bytes_be())),
             Arg::Value(DeprecatedFelt::from_bytes_be(&b.clone().to_bytes_be())),
//            Arg::Value(DeprecatedFelt::from_bytes_be(&c.clone().to_bytes_be()))],
            Arg::Value(DeprecatedFelt::from_bytes_be(&c.clone().to_bytes_be()))],
//            Some(DEFAULT_GAS as usize),
            Some(DEFAULT_GAS as usize),
//        )
        )
//        .unwrap();
        .unwrap();
//

//        let result_native = run_native_program(
        let result_native = run_native_program(
//            &POSEIDON,
            &POSEIDON,
//            "run_test",
            "run_test",
//            &[JitValue::Felt252(a), JitValue::Felt252(b), JitValue::Felt252(c)],
            &[JitValue::Felt252(a), JitValue::Felt252(b), JitValue::Felt252(c)],
//            Some(DEFAULT_GAS as u128),
            Some(DEFAULT_GAS as u128),
//            Option::<DummySyscallHandler>::None,
            Option::<DummySyscallHandler>::None,
//        );
        );
//

//        compare_outputs(
        compare_outputs(
//            &POSEIDON.1,
            &POSEIDON.1,
//            &POSEIDON.2.find_function("run_test").unwrap().id,
            &POSEIDON.2.find_function("run_test").unwrap().id,
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
//

//#[test]
#[test]
//fn self_referencing_struct() {
fn self_referencing_struct() {
//    let result_vm = run_vm_program(
    let result_vm = run_vm_program(
//        &SELF_REFERENCING,
        &SELF_REFERENCING,
//        "run_test",
        "run_test",
//        &[],
        &[],
//        Some(DEFAULT_GAS as usize),
        Some(DEFAULT_GAS as usize),
//    )
    )
//    .unwrap();
    .unwrap();
//    let result_native = run_native_program(
    let result_native = run_native_program(
//        &SELF_REFERENCING,
        &SELF_REFERENCING,
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
//        &SELF_REFERENCING.1,
        &SELF_REFERENCING.1,
//        &SELF_REFERENCING.2.find_function("run_test").unwrap().id,
        &SELF_REFERENCING.2.find_function("run_test").unwrap().id,
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
//fn no_op() {
fn no_op() {
//    let result_vm = run_vm_program(&NO_OP, "run_test", &[], Some(DEFAULT_GAS as usize)).unwrap();
    let result_vm = run_vm_program(&NO_OP, "run_test", &[], Some(DEFAULT_GAS as usize)).unwrap();
//    let result_native = run_native_program(
    let result_native = run_native_program(
//        &NO_OP,
        &NO_OP,
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
//        &NO_OP.1,
        &NO_OP.1,
//        &NO_OP.2.find_function("run_test").unwrap().id,
        &NO_OP.2.find_function("run_test").unwrap().id,
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
