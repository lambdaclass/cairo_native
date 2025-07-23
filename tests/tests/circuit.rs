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
            CircuitData, CircuitElement, CircuitInput, circuit_add, circuit_sub, circuit_mul, circuit_inverse,
            EvalCircuitResult, EvalCircuitTrait, u384, CircuitOutputsTrait, CircuitModulus, into_u96_guarantee, U96Guarantee,
            CircuitInputs, AddInputResultTrait, AddInputResult, IntoCircuitInputValue, add_circuit_input
        };

        #[feature("bounded-int-utils")]
        use core::internal::bounded_int::BoundedInt;

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

        fn test_circuit_fail() -> u384 {
            let in1 = CircuitElement::<CircuitInput<0>> {};
            let in2 = CircuitElement::<CircuitInput<1>> {};
            let add = circuit_add(in1, in2);

            let modulus = TryInto::<_, CircuitModulus>::try_into([0, 0, 0, 0]).unwrap(); // Having this modulus makes eval panic

            let outputs = (add,)
                .new_inputs()
                .next([3, 3, 3, 3])
                .next([6, 6, 6, 6])
                .done()
                .eval(modulus)
                .unwrap();

            outputs.get_output(add)
        }

        fn test_into_u96_guarantee() -> (U96Guarantee, U96Guarantee, U96Guarantee) {
            (
                into_u96_guarantee::<BoundedInt<0, 79228162514264337593543950335>>(123),
                into_u96_guarantee::<BoundedInt<100, 1000>>(123),
                into_u96_guarantee::<u8>(123),
            )
        }


        #[generate_trait]
        pub impl AddInputResultImpl2<C> of AddInputResultTrait2<C> {
            fn next_2<Value, +IntoCircuitInputValue<Value>, +Drop<Value>>(
                self: AddInputResult<C>, value: Value,
            ) -> AddInputResult<C> {
                match self {
                    AddInputResult::More(accumulator) => add_circuit_input(
                        accumulator, value.into_circuit_input_value(),
                    ),
                    AddInputResult::Done(_) => panic!("All inputs have been filled"),
                }
            }
            #[inline(always)]
            fn done_2(self: AddInputResult<C>) -> CircuitData<C> {
                match self {
                    AddInputResult::Done(data) => data,
                    AddInputResult::More(_) => panic!("not all inputs filled"),
                }
            }
        }

        // Returns the modulus of BN254
        #[inline(always)]
        fn get_BN254_modulus() -> CircuitModulus {
            let modulus = TryInto::<
                _, CircuitModulus,
            >::try_into([0x6871ca8d3c208c16d87cfd47, 0xb85045b68181585d97816a91, 0x30644e72e131a029, 0x0])
                .unwrap();
            modulus
        }

        fn compute_yInvXnegOverY_BN254() -> (u384, u384) {
            let in1 = CircuitElement::<CircuitInput<0>> {};
            let in2 = CircuitElement::<CircuitInput<1>> {};
            let in3 = CircuitElement::<CircuitInput<2>> {};
            let yInv = circuit_inverse(in3);
            let xNeg = circuit_sub(in1, in2);
            let xNegOverY = circuit_mul(xNeg, yInv);

            let modulus = get_BN254_modulus(); // BN254 prime field modulus

            let outputs = (yInv, xNegOverY)
                .new_inputs()
                .next_2([0, 0, 0, 0])
                .next_2([7, 0, 0, 0])
                .next_2([3, 0, 0, 0])
                .done_2()
                .eval(modulus)
                .unwrap();

            return (outputs.get_output(yInv), outputs.get_output(xNegOverY));
        }

        #[inline(always)]
        pub fn batch_3_mod_bn254() -> u384 {
            let _x = CircuitElement::<CircuitInput<0>> {};
            let _y = CircuitElement::<CircuitInput<1>> {};
            let _z = CircuitElement::<CircuitInput<2>> {};
            let _c0 = CircuitElement::<CircuitInput<3>> {};
            let _c1 = circuit_mul(_c0, _c0);
            let _c2 = circuit_mul(_c1, _c0);
            let _mul1 = circuit_mul(_x, _c0);
            let _mul2 = circuit_mul(_y, _c1);
            let _mul3 = circuit_mul(_z, _c2);
            let res = circuit_add(circuit_add(_mul1, _mul2), _mul3);

            let modulus = get_BN254_modulus(); // BN254 prime field modulus

            let outputs = (res,)
                .new_inputs()
                .next_2([5, 0, 0, 0])
                .next_2([2, 0, 0, 0])
                .next_2([8, 0, 0, 0])
                .next_2([3, 0, 0, 0])
                .done_2()
                .eval(modulus)
                .unwrap();

            return outputs.get_output(res);
        }

        #[derive(Copy, Drop, Debug, PartialEq)]
        pub struct G2Point {
            pub x0: u384,
            pub x1: u384,
            pub y0: u384,
            pub y1: u384,
        }

        #[inline(always)]
        pub fn run_ADD_EC_POINTS_G2_circuit() -> (G2Point,) {
            let p = G2Point {
                x0: u384 {limb0: 5, limb1: 0, limb2: 0, limb3: 0},
                x1: u384 {limb0: 1, limb1: 0, limb2: 0, limb3: 0},
                y0: u384 {limb0: 7, limb1: 0, limb2: 0, limb3: 0},
                y1: u384 {limb0: 2, limb1: 0, limb2: 0, limb3: 0},
            };

            let q = G2Point {
                x0: u384 {limb0: 3, limb1: 0, limb2: 0, limb3: 0},
                x1: u384 {limb0: 4, limb1: 0, limb2: 0, limb3: 0},
                y0: u384 {limb0: 6, limb1: 0, limb2: 0, limb3: 0},
                y1: u384 {limb0: 8, limb1: 0, limb2: 0, limb3: 0},
            };

            // CONSTANT stack
            let in0 = CircuitElement::<CircuitInput<0>> {}; // 0x0

            // INPUT stack
            let (in1, in2, in3) = (CircuitElement::<CircuitInput<1>> {}, CircuitElement::<CircuitInput<2>> {}, CircuitElement::<CircuitInput<3>> {});
            let (in4, in5, in6) = (CircuitElement::<CircuitInput<4>> {}, CircuitElement::<CircuitInput<5>> {}, CircuitElement::<CircuitInput<6>> {});
            let (in7, in8) = (CircuitElement::<CircuitInput<7>> {}, CircuitElement::<CircuitInput<8>> {});
            let t0 = circuit_sub(in3, in7); // Fp2 sub coeff 0/1
            let t1 = circuit_sub(in4, in8); // Fp2 sub coeff 1/1
            let t2 = circuit_sub(in1, in5); // Fp2 sub coeff 0/1
            let t3 = circuit_sub(in2, in6); // Fp2 sub coeff 1/1
            let t4 = circuit_mul(t2, t2); // Fp2 Inv start
            let t5 = circuit_mul(t3, t3);
            let t6 = circuit_add(t4, t5);
            let t7 = circuit_inverse(t6);
            let t8 = circuit_mul(t2, t7); // Fp2 Inv real part end
            let t9 = circuit_mul(t3, t7);
            let t10 = circuit_sub(in0, t9); // Fp2 Inv imag part end
            let t11 = circuit_mul(t0, t8); // Fp2 mul start
            let t12 = circuit_mul(t1, t10);
            let t13 = circuit_sub(t11, t12); // Fp2 mul real part end
            let t14 = circuit_mul(t0, t10);
            let t15 = circuit_mul(t1, t8);
            let t16 = circuit_add(t14, t15); // Fp2 mul imag part end
            let t17 = circuit_add(t13, t16);
            let t18 = circuit_sub(t13, t16);
            let t19 = circuit_mul(t17, t18);
            let t20 = circuit_mul(t13, t16);
            let t21 = circuit_add(t20, t20);
            let t22 = circuit_sub(t19, in1); // Fp2 sub coeff 0/1
            let t23 = circuit_sub(t21, in2); // Fp2 sub coeff 1/1
            let t24 = circuit_sub(t22, in5); // Fp2 sub coeff 0/1
            let t25 = circuit_sub(t23, in6); // Fp2 sub coeff 1/1
            let t26 = circuit_sub(in1, t24); // Fp2 sub coeff 0/1
            let t27 = circuit_sub(in2, t25); // Fp2 sub coeff 1/1
            let t28 = circuit_mul(t13, t26); // Fp2 mul start
            let t29 = circuit_mul(t16, t27);
            let t30 = circuit_sub(t28, t29); // Fp2 mul real part end
            let t31 = circuit_mul(t13, t27);
            let t32 = circuit_mul(t16, t26);
            let t33 = circuit_add(t31, t32); // Fp2 mul imag part end
            let t34 = circuit_sub(t30, in3); // Fp2 sub coeff 0/1
            let t35 = circuit_sub(t33, in4); // Fp2 sub coeff 1/1

            let modulus = get_BN254_modulus();

            let mut circuit_inputs = (t24, t25, t34, t35).new_inputs();
            // Prefill constants:
            circuit_inputs = circuit_inputs.next_2([0x0, 0x0, 0x0, 0x0]); // in0
            // Fill inputs:
            circuit_inputs = circuit_inputs.next_2(p.x0); // in1
            circuit_inputs = circuit_inputs.next_2(p.x1); // in2
            circuit_inputs = circuit_inputs.next_2(p.y0); // in3
            circuit_inputs = circuit_inputs.next_2(p.y1); // in4
            circuit_inputs = circuit_inputs.next_2(q.x0); // in5
            circuit_inputs = circuit_inputs.next_2(q.x1); // in6
            circuit_inputs = circuit_inputs.next_2(q.y0); // in7
            circuit_inputs = circuit_inputs.next_2(q.y1); // in8

            let outputs = circuit_inputs.done_2().eval(modulus).unwrap();
            let result: G2Point = G2Point {
                x0: outputs.get_output(t24),
                x1: outputs.get_output(t25),
                y0: outputs.get_output(t34),
                y1: outputs.get_output(t35),
            };
            return (result,);
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
fn comparison_circuit_add() {
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
fn comparison_circuit_sub() {
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
fn comparison_circuit_mul() {
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
fn comparison_circuit_inv() {
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
fn comparison_circuit_full() {
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

#[test]
fn comparison_circuit_fail() {
    let program = &TEST;

    let result_vm = run_vm_program(
        program,
        "test_circuit_fail",
        vec![],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();

    let result_native = run_native_program(
        program,
        "test_circuit_fail",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program.2.find_function("test_circuit_fail").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn comparison_circuit_into_u96_guarantee() {
    let program = &TEST;

    let result_vm = run_vm_program(
        program,
        "test_into_u96_guarantee",
        vec![],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();

    let result_native = run_native_program(
        program,
        "test_into_u96_guarantee",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program
            .2
            .find_function("test_into_u96_guarantee")
            .unwrap()
            .id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn comparison_circuit_y_inv_x_neg_over_y_bn254() {
    let program = &TEST;

    let result_vm = run_vm_program(
        program,
        "compute_yInvXnegOverY_BN254",
        vec![],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();

    let result_native = run_native_program(
        program,
        "compute_yInvXnegOverY_BN254",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program
            .2
            .find_function("compute_yInvXnegOverY_BN254")
            .unwrap()
            .id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn comparison_batch_3_mod_bn254() {
    let program = &TEST;

    let result_vm = run_vm_program(
        program,
        "batch_3_mod_bn254",
        vec![],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();

    let result_native = run_native_program(
        program,
        "batch_3_mod_bn254",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program.2.find_function("batch_3_mod_bn254").unwrap().id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn test_circuit_add_ec_points_g2() {
    let program = &TEST;

    let result_vm = run_vm_program(
        program,
        "run_ADD_EC_POINTS_G2_circuit",
        vec![],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();

    let result_native = run_native_program(
        program,
        "run_ADD_EC_POINTS_G2_circuit",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program
            .2
            .find_function("run_ADD_EC_POINTS_G2_circuit")
            .unwrap()
            .id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}
