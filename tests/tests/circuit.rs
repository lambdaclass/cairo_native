use crate::common::{compare_outputs, DEFAULT_GAS};
use crate::common::{load_cairo, run_native_program, run_vm_program};
use cairo_lang_runner::SierraCasmRunner;
use cairo_lang_sierra::program::Program;
use cairo_native::starknet::DummySyscallHandler;
use cairo_native::Value;
use lazy_static::lazy_static;

lazy_static! {
    // Taken from: https://github.com/starkware-libs/sequencer/blob/7ee6f4c8a81def87402c626c9d72a33c74bc3243/crates/blockifier/feature_contracts/cairo1/test_contract.cairo#L656
    static ref TEST: (String, Program, SierraCasmRunner) = load_cairo! {
        use core::circuit::{
            CircuitElement, CircuitInput, circuit_add, circuit_sub, circuit_mul, circuit_inverse,
            EvalCircuitResult, EvalCircuitTrait, u384, CircuitOutputsTrait, CircuitModulus,
            CircuitInputs, AddInputResultTrait
        };

        fn test_guarantee_first_limb() {
            let in1 = CircuitElement::<CircuitInput<0>> {};
            let in2 = CircuitElement::<CircuitInput<1>> {};
            let add = circuit_add(in1, in2);
            let inv = circuit_inverse(add);
            let sub = circuit_sub(inv, in2);
            let mul = circuit_mul(inv, sub);

            let modulus = TryInto::<_, CircuitModulus>::try_into([7, 0, 0, 0]).unwrap();
            let outputs = (mul,)
                .new_inputs()
                .next([3, 0, 0, 0])
                .next([6, 0, 0, 0])
                .done()
                .eval(modulus)
                .unwrap();

            assert!(outputs.get_output(mul) == u384 { limb0: 6, limb1: 0, limb2: 0, limb3: 0 });
        }

        fn test_guarantee_last_limb() {
            let in1 = CircuitElement::<CircuitInput<0>> {};
            let in2 = CircuitElement::<CircuitInput<1>> {};
            let add = circuit_add(in1, in2);

            let modulus = TryInto::<_, CircuitModulus>::try_into([7, 0, 0, 1]).unwrap();
            let outputs = (add,)
                .new_inputs()
                .next([5, 0, 0, 0])
                .next([9, 0, 0, 0])
                .done()
                .eval(modulus)
                .unwrap();

            assert!(outputs.get_output(add) == u384 { limb0: 14, limb1: 0, limb2: 0, limb3: 0 });
        }

        fn test_guarantee_middle_limb() {
            let in1 = CircuitElement::<CircuitInput<0>> {};
            let in2 = CircuitElement::<CircuitInput<1>> {};
            let add = circuit_add(in1, in2);

            let modulus = TryInto::<_, CircuitModulus>::try_into([7, 0, 1, 0]).unwrap();
            let outputs = (add,)
                .new_inputs()
                .next([5, 0, 0, 0])
                .next([9, 0, 0, 0])
                .done()
                .eval(modulus)
                .unwrap();

            assert!(outputs.get_output(add) == u384 { limb0: 14, limb1: 0, limb2: 0, limb3: 0 });
        }

        fn test_circuit_add() -> u384 {
            let in1 = CircuitElement::<CircuitInput<0>> {};
            let in2 = CircuitElement::<CircuitInput<1>> {};
            let add = circuit_add(in1, in2);

            let modulus = TryInto::<_, CircuitModulus>::try_into([12, 12, 12, 12]).unwrap();

            let outputs = (add,)
                .new_inputs()
                .next([3, 3, 3, 3])
                .next([6, 6, 6, 6])
                .done()
                .eval(modulus)
                .unwrap();

            outputs.get_output(add)
        }

        fn test_circuit_sub() -> u384 {
            let in1 = CircuitElement::<CircuitInput<0>> {};
            let in2 = CircuitElement::<CircuitInput<1>> {};
            let sub = circuit_sub(in1, in2);

            let modulus = TryInto::<_, CircuitModulus>::try_into([12, 12, 12, 12]).unwrap();

            let outputs = (sub,)
                .new_inputs()
                .next([6, 6, 6, 6])
                .next([3, 3, 3, 3])
                .done()
                .eval(modulus)
                .unwrap();

            outputs.get_output(sub)
        }

        fn test_circuit_mul() -> u384 {
            let in1 = CircuitElement::<CircuitInput<0>> {};
            let in2 = CircuitElement::<CircuitInput<1>> {};
            let mul = circuit_mul(in1, in2);

            let modulus = TryInto::<_, CircuitModulus>::try_into([12, 12, 12, 12]).unwrap();

            let outputs = (mul,)
                .new_inputs()
                .next([3, 0, 0, 0])
                .next([3, 3, 3, 3])
                .done()
                .eval(modulus)
                .unwrap();

            outputs.get_output(mul)
        }

        fn test_circuit_inv() -> u384 {
            let in1 = CircuitElement::<CircuitInput<0>> {};
            let inv = circuit_inverse(in1);

            let modulus = TryInto::<_, CircuitModulus>::try_into([11, 0, 0, 0]).unwrap();

            let outputs = (inv,)
                .new_inputs()
                .next([2, 0, 0, 0])
                .done()
                .eval(modulus)
                .unwrap();

            outputs.get_output(inv)
        }

        fn test_circuit_full() -> u384 {
            let in1 = CircuitElement::<CircuitInput<0>> {};
            let in2 = CircuitElement::<CircuitInput<1>> {};
            let add1 = circuit_add(in1, in2);
            let mul1 = circuit_mul(add1, in1);
            let mul2 = circuit_mul(mul1, add1);
            let inv1 = circuit_inverse(mul2);
            let sub1 = circuit_sub(inv1, in2);
            let sub2 = circuit_sub(sub1, mul2);
            let inv2 = circuit_inverse(sub2);
            let add2 = circuit_add(inv2, inv2);

            let modulus = TryInto::<_, CircuitModulus>::try_into([17, 14, 14, 14]).unwrap();

            let outputs = (add2,)
                .new_inputs()
                .next([9, 2, 9, 3])
                .next([5, 7, 0, 8])
                .done()
                .eval(modulus)
                .unwrap();

            outputs.get_output(add2)
        }


    };

}

#[test]
fn circuit_guarantee_first_limb() {
    let program = &TEST;

    let result_vm = run_vm_program(
        program,
        "test_guarantee_first_limb",
        vec![],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();

    let result_native = run_native_program(
        program,
        "test_guarantee_first_limb",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    assert!(matches!(
        result_native.return_value,
        Value::Enum { tag: 0, .. }
    ));

    compare_outputs(
        &program.1,
        &program
            .2
            .find_function("test_guarantee_first_limb")
            .unwrap()
            .id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn circuit_guarantee_last_limb() {
    let program = &TEST;

    let result_vm = run_vm_program(
        program,
        "test_guarantee_last_limb",
        vec![],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();

    let result_native = run_native_program(
        program,
        "test_guarantee_last_limb",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    assert!(matches!(
        result_native.return_value,
        Value::Enum { tag: 0, .. }
    ));

    compare_outputs(
        &program.1,
        &program
            .2
            .find_function("test_guarantee_last_limb")
            .unwrap()
            .id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn circuit_guarantee_middle_limb() {
    let program = &TEST;

    let result_vm = run_vm_program(
        program,
        "test_guarantee_middle_limb",
        vec![],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();

    let result_native = run_native_program(
        program,
        "test_guarantee_middle_limb",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    assert!(matches!(
        result_native.return_value,
        Value::Enum { tag: 0, .. }
    ));

    compare_outputs(
        &program.1,
        &program
            .2
            .find_function("test_guarantee_middle_limb")
            .unwrap()
            .id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn builtin_comparison_circuit_add() {
    let program = &TEST;

    let result_vm = run_vm_program(
        program,
        "test_circuit_add",
        vec![],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();

    let result_native = run_native_program(
        program,
        "test_circuit_add",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program.2.find_function("test_circuit_add").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn builtin_comparison_circuit_sub() {
    let program = &TEST;

    let result_vm = run_vm_program(
        program,
        "test_circuit_sub",
        vec![],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();

    let result_native = run_native_program(
        program,
        "test_circuit_sub",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program.2.find_function("test_circuit_sub").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn builtin_comparison_circuit_mul() {
    let program = &TEST;

    let result_vm = run_vm_program(
        program,
        "test_circuit_mul",
        vec![],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();

    let result_native = run_native_program(
        program,
        "test_circuit_mul",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program.2.find_function("test_circuit_mul").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn builtin_comparison_circuit_inv() {
    let program = &TEST;

    let result_vm = run_vm_program(
        program,
        "test_circuit_inv",
        vec![],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();

    let result_native = run_native_program(
        program,
        "test_circuit_inv",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program.2.find_function("test_circuit_inv").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn builtin_comparison_circuit_full() {
    let program = &TEST;

    let result_vm = run_vm_program(
        program,
        "test_circuit_full",
        vec![],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();

    let result_native = run_native_program(
        program,
        "test_circuit_full",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program.2.find_function("test_circuit_full").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}
