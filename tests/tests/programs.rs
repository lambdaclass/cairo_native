use crate::common::{any_felt, load_cairo, run_native_program, run_vm_program};
use crate::common::{compare_outputs, DEFAULT_GAS};
use cairo_lang_runner::{Arg, SierraCasmRunner};
use cairo_lang_sierra::program::Program;
use cairo_native::starknet::DummySyscallHandler;
use cairo_native::utils::felt252_str;
use cairo_native::Value;
use lazy_static::lazy_static;
use proptest::prelude::*;
use starknet_types_core::felt::Felt;

lazy_static! {
    pub static ref FACTORIAL: (String, Program, SierraCasmRunner) = load_cairo! {
        fn factorial(value: felt252, n: felt252) -> felt252 {
            if (n == 1) {
                value
            } else {
                factorial(value * n, n - 1)
            }
        }

        fn run_test(n: felt252) -> felt252 {
            factorial(1, n)
        }
    };

    pub static ref FIB: (String, Program, SierraCasmRunner) = load_cairo! {
        fn fib(a: felt252, b: felt252, n: felt252) -> felt252 {
            match n {
                0 => a,
                _ => fib(b, a + b, n - 1),
            }
        }

        fn run_test(n: felt252) -> felt252 {
            fib(0, 1, n)
        }
    };

    pub static ref LOGISTIC_MAP: (String, Program, SierraCasmRunner) = load_cairo! {
        fn iterate_map(r: felt252, x: felt252) -> felt252 {
            r * x * -x
        }

        // good default: 1000
        fn run_test(mut i: felt252) -> felt252 {
            // Initial value.
            let mut x = 1234567890123456789012345678901234567890;

            // Iterate the map.
            loop {
                x = iterate_map(4, x);

                if i == 0 {
                    break x;
                }

                i = i - 1;
            }
        }
    };

    pub static ref PEDERSEN: (String, Program, SierraCasmRunner) = load_cairo! {
        use core::pedersen::pedersen;

        fn run_test(a: felt252, b: felt252) -> felt252 {
            pedersen(a, b)
        }
    };

    pub static ref POSEIDON: (String, Program, SierraCasmRunner) = load_cairo! {
        use core::poseidon::hades_permutation;

        fn run_test(a: felt252, b: felt252, c: felt252) -> (felt252, felt252, felt252) {
            hades_permutation(a, b, c)
        }
    };

    pub static ref SELF_REFERENCING: (String, Program, SierraCasmRunner) = load_cairo! {
        #[derive(Drop, Copy, PartialEq)]
        enum ArrayItem {
            Span: Span<u8>,
            Recursive: Span<ArrayItem>
        }

        fn recursion(input: Span<u8>) -> Span<ArrayItem> {
            let mut output: Array<ArrayItem> = Default::default();

            let index = (*input.at(0));
            if index < 5 {
                output.append(ArrayItem::Span(input));
            } else {
                let res = recursion(input.slice(1, input.len() - 1));
                output.append(ArrayItem::Recursive(res));
            }

            return output.span();
        }

        fn run_test() -> Span<ArrayItem> {
            let arr = array![10, 9, 8, 7, 6, 4];
            recursion(arr.span())
        }
    };

    pub static ref NO_OP: (String, Program, SierraCasmRunner) = load_cairo! {
        #[inline(never)]
        fn no_op() {}

        fn run_test() {
            no_op();
        }
    };
}

#[test]
fn fib() {
    let result_vm = run_vm_program(
        &FIB,
        "run_test",
        vec![Arg::Value(Felt::from(10))],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();
    let result_native = run_native_program(
        &FIB,
        "run_test",
        &[Value::Felt252(10.into())],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &FIB.1,
        &FIB.2.find_function("run_test").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn logistic_map() {
    let result_vm = run_vm_program(
        &LOGISTIC_MAP,
        "run_test",
        vec![Arg::Value(Felt::from(1000))],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();
    let result_native = run_native_program(
        &LOGISTIC_MAP,
        "run_test",
        &[Value::Felt252(1000.into())],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &LOGISTIC_MAP.1,
        &LOGISTIC_MAP.2.find_function("run_test").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn pedersen() {
    let result_vm = run_vm_program(
        &PEDERSEN,
        "run_test",
        vec![
            Arg::Value(
                Felt::from_dec_str(
                    "2163739901324492107409690946633517860331020929182861814098856895601180685",
                )
                .unwrap(),
            ),
            Arg::Value(
                Felt::from_dec_str(
                    "2392090257937917229310563411601744459500735555884672871108624696010915493156",
                )
                .unwrap(),
            ),
        ],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();
    let result_native = run_native_program(
        &PEDERSEN,
        "run_test",
        &[
            Value::Felt252(felt252_str(
                "2163739901324492107409690946633517860331020929182861814098856895601180685",
            )),
            Value::Felt252(felt252_str(
                "2392090257937917229310563411601744459500735555884672871108624696010915493156",
            )),
        ],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &PEDERSEN.1,
        &PEDERSEN.2.find_function("run_test").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn factorial() {
    let result_vm = run_vm_program(
        &FACTORIAL,
        "run_test",
        vec![Arg::Value(Felt::from(13))],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();
    let result_native = run_native_program(
        &FACTORIAL,
        "run_test",
        &[Value::Felt252(13.into())],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &FACTORIAL.1,
        &FACTORIAL.2.find_function("run_test").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

proptest! {
    #[test]
    fn fib_proptest(n in 0..100i32) {
        let result_vm = run_vm_program(
            &FIB,
            "run_test",
            vec![Arg::Value(Felt::from(n))],
            Some(DEFAULT_GAS as usize),
        )
        .unwrap();
        let result_native = run_native_program(
            &FIB,
            "run_test",
            &[Value::Felt252(n.into())],
            Some(DEFAULT_GAS),
            Option::<DummySyscallHandler>::None,
        );

        compare_outputs(
            &FIB.1,
            &FIB.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    #[test]
    fn logistic_map_proptest(n in 100..110i32) {
        let result_vm = run_vm_program(
            &LOGISTIC_MAP,
            "run_test",
            vec![Arg::Value(Felt::from(n))],
            Some(DEFAULT_GAS as usize),
        )
        .unwrap();
        let result_native = run_native_program(
            &LOGISTIC_MAP,
            "run_test",
            &[Value::Felt252(n.into())],
            Some(DEFAULT_GAS),
            Option::<DummySyscallHandler>::None,
        );

        compare_outputs(
            &LOGISTIC_MAP.1,
            &LOGISTIC_MAP.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    #[test]
    fn factorial_proptest(n in 1..100i32) {
        let result_vm = run_vm_program(
            &FACTORIAL,
            "run_test",
            vec![Arg::Value(Felt::from(n))],
            Some(DEFAULT_GAS as usize),
        )
        .unwrap();
        let result_native = run_native_program(
            &FACTORIAL,
            "run_test",
            &[Value::Felt252(n.into())],
            Some(DEFAULT_GAS),
            Option::<DummySyscallHandler>::None,
        );

        compare_outputs(
            &FACTORIAL.1,
            &FACTORIAL.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    #[test]
    fn pedersen_proptest(a in any_felt(), b in any_felt()) {
        let result_vm = run_vm_program(
            &PEDERSEN,
            "run_test",
            vec![Arg::Value(Felt::from_bytes_be(&a.clone().to_bytes_be())), Arg::Value(Felt::from_bytes_be(&b.clone().to_bytes_be()))],
            Some(DEFAULT_GAS as usize),
        )
        .unwrap();

        let result_native = run_native_program(
            &PEDERSEN,
            "run_test",
            &[Value::Felt252(a), Value::Felt252(b)],
            Some(DEFAULT_GAS),
            Option::<DummySyscallHandler>::None,
        );

        compare_outputs(
            &PEDERSEN.1,
            &PEDERSEN.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    #[test]
    fn poseidon_proptest(a in any_felt(), b in any_felt(), c in any_felt()) {
        let result_vm = run_vm_program(
            &POSEIDON,
            "run_test",
            vec![Arg::Value(Felt::from_bytes_be(&a.clone().to_bytes_be())),
             Arg::Value(Felt::from_bytes_be(&b.clone().to_bytes_be())),
            Arg::Value(Felt::from_bytes_be(&c.clone().to_bytes_be()))],
            Some(DEFAULT_GAS as usize),
        )
        .unwrap();

        let result_native = run_native_program(
            &POSEIDON,
            "run_test",
            &[Value::Felt252(a), Value::Felt252(b), Value::Felt252(c)],
            Some(DEFAULT_GAS),
            Option::<DummySyscallHandler>::None,
        );

        compare_outputs(
            &POSEIDON.1,
            &POSEIDON.2.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }
}

#[test]
fn self_referencing_struct() {
    let result_vm = run_vm_program(
        &SELF_REFERENCING,
        "run_test",
        vec![],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();
    let result_native = run_native_program(
        &SELF_REFERENCING,
        "run_test",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &SELF_REFERENCING.1,
        &SELF_REFERENCING.2.find_function("run_test").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn no_op() {
    let result_vm = run_vm_program(&NO_OP, "run_test", vec![], Some(DEFAULT_GAS as usize)).unwrap();
    let result_native = run_native_program(
        &NO_OP,
        "run_test",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &NO_OP.1,
        &NO_OP.2.find_function("run_test").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}
