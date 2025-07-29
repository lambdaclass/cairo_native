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

    };
}

lazy_static! {
    static ref GARAGA_CIRCUITS: (String, Program, SierraCasmRunner) = load_cairo! {
        use core::circuit::{
            CircuitData, CircuitElement, CircuitInput, circuit_add, circuit_sub, circuit_mul, circuit_inverse,
            EvalCircuitResult, EvalCircuitTrait, u384, CircuitOutputsTrait, CircuitModulus, into_u96_guarantee, U96Guarantee,
            CircuitInputs, AddInputResultTrait, AddInputResult, IntoCircuitInputValue, add_circuit_input
        };

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

        // Taken from: https://github.com/keep-starknet-strange/garaga/blob/5c5859e6dc5515f542c310cb38a149602e774112/src/src/definitions/curves.cairo#L336
        // Returns the modulus of BN254
        #[inline(always)]
        fn get_BN254_modulus() -> CircuitModulus {
            let modulus = TryInto::<
                _, CircuitModulus,
            >::try_into([0x6871ca8d3c208c16d87cfd47, 0xb85045b68181585d97816a91, 0x30644e72e131a029, 0x0])
                .unwrap();
            modulus
        }

        // Taken from: https://github.com/keep-starknet-strange/garaga/blob/5c5859e6dc5515f542c310cb38a149602e774112/src/src/basic_field_ops.cairo#L60
        // Computes (in1 - in2) * (in3 ** -1)
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
                .next_2([
                    0xae40a8b5aee95e54aedee2e7,
                    0x6e0699501c5035eed8fc5162,
                    0xbee76829b76806d1b6617bf8,
                    0x5026c3305c1267922077393,
                ])
                .next_2([
                    0x10c08c4b0a70e02491c3c435,
                    0x591ef738050b3ce067e2016f,
                    0xdd6e0a179e2ce3c1399c5273,
                    0xd5c9af9b97e94f90cb4aba3,
                ])
                .next_2([
                    0x93be53660cebb92c90d4fa87,
                    0xfbf63ca94e1d0ffd65801863,
                    0xd24fd9a06d72f1dc57f15f0a,
                    0x100dbfd4f271378e85171313,
                ])
                .done_2()
                .eval(modulus)
                .unwrap();

            return (outputs.get_output(yInv), outputs.get_output(xNegOverY));
        }

        // Taken from: https://github.com/keep-starknet-strange/garaga/blob/5c5859e6dc5515f542c310cb38a149602e774112/src/src/basic_field_ops.cairo#L174
        // In the original function, the modulus is a parameter. Here we will use BN254 modulus.
        // Computes _x * _c0 + _y * _c0 ** 2 + _z * _c0 ** 3
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
                .next_2([
                    0xb7296e587409163eecd3ef5d,
                    0x8a065d6871fa185d15703e78,
                    0x8a85fb95bb90eb5c7a0d81a9,
                    0x157cf362e91a3c96640bd973
                ])
                .next_2([
                    0x2131be4b061714de5a11407d,
                    0xd41318f9bcade1fee985310b,
                    0xb2669e638a7b78b7ba5c6751,
                    0xa5284fb2911d4e2f445e714,
                ])
                .next_2([
                    0x712edcaf95ed642a8237e6fd,
                    0xed6fccd7b64896ebb6ffb3d9,
                    0xfcb88d23294a46657b8d2482,
                    0x143ef485b660d37036fc18e2,
                ])
                .next_2([
                    0xaa5b7ff57bdbf47e6ab49121,
                    0xc14cded56b4a44e022320616,
                    0xdd5105feb3fdc5b10edb5afa,
                    0x175d2c78538490ce02fcead8,
                ])
                .done_2()
                .eval(modulus)
                .unwrap();

            return outputs.get_output(res);
        }

        // Taken from: https://github.com/keep-starknet-strange/garaga/blob/5c5859e6dc5515f542c310cb38a149602e774112/src/src/definitions/structs/points.cairo#L27
        // Represents a point on G2, the group of rational points on an elliptic curve over an extension field.
        #[derive(Copy, Drop, Debug, PartialEq)]
        pub struct G2Point {
            pub x0: u384,
            pub x1: u384,
            pub y0: u384,
            pub y1: u384,
        }

        // Taken from: https://github.com/keep-starknet-strange/garaga/blob/5c5859e6dc5515f542c310cb38a149602e774112/src/src/circuits/ec.cairo#L324
        // Adds 2 ec G2Points without checking if:
        //  - They are on the curve
        //  - They are on infinity (same x but opposite y)
        #[inline(always)]
        pub fn run_ADD_EC_POINTS_G2_circuit() -> (G2Point,) {
            let p = G2Point {
                x0: u384 {
                    limb0: 0xf3611b78c952aacab827a053,
                    limb1: 0xe1ea1e1e4d00dbae81f14b0b,
                    limb2: 0xcc7ed5863bc0b995b8825e0e,
                    limb3: 0x1638533957d540a9d2370f17,
                },
                x1: u384 {
                    limb0: 0xb57ec72a6178288c47c33577,
                    limb1: 0x728114d1031e1572c6c886f6,
                    limb2: 0x730a124fd70662a904ba1074,
                    limb3: 0xa4edef9c1ed7f729f520e47,
                },
                y0: u384 {
                    limb0: 0x764bf3bd999d95d71e4c9899,
                    limb1: 0xbfe6bd221e47aa8ae88dece9,
                    limb2: 0x2b5256789a66da69bf91009c,
                    limb3: 0x468fb440d82b0630aeb8dca,
                },
                y1: u384 {
                    limb0: 0xa59c8967acdefd8b6e36ccf3,
                    limb1: 0x97003f7a13c308f5422e1aa0,
                    limb2: 0x3f887136a43253d9c66c4116,
                    limb3: 0xf6d4552fa65dd2638b36154,
                },
            };

            let q = G2Point {
                x0: u384 {
                    limb0: 0x866f09d516020ef82324afae,
                    limb1: 0xa0c75df1c04d6d7a50a030fc,
                    limb2: 0xdccb23ae691ae54329781315,
                    limb3: 0x122915c824a0857e2ee414a3,
                },
                x1: u384 {
                    limb0: 0x937cc6d9d6a44aaa56ca66dc,
                    limb1: 0x5062650f8d251c96eb480673,
                    limb2: 0x7e0550ff2ac480905396eda5,
                    limb3: 0x9380275bbc8e5dcea7dc4dd,
                },
                y0: u384 {
                    limb0: 0x8b52fdf2455e44813ecfd892,
                    limb1: 0x326ac738fef5c721479dfd94,
                    limb2: 0xbc1a6f0136961d1e3b20b1a7,
                    limb3: 0xb21da7955969e61010c7a1a,
                },
                y1: u384 {
                    limb0: 0xb975b9edea56d53f23a0e849,
                    limb1: 0x714150a166bfbd6bcf6b3b58,
                    limb2: 0xa36cfe5f62a7e42e0bf1c1ed,
                    limb3: 0x8f239ba329b3967fe48d718,
                },
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

lazy_static! {
    // Taken from: https://github.com/kkrt-labs/kakarot/blob/563af42d5fe9888f8f49cf22003d2085612bf42c/cairo/kakarot-ssj/crates/evm/src/precompiles/ec_operations/ec_add.cairo#L143
    static ref KAKAROT_CIRCUIT: (String, Program, SierraCasmRunner) = load_cairo! {
        use core::circuit::{
            CircuitData, CircuitElement, CircuitInput, circuit_add, circuit_sub, circuit_mul, circuit_inverse,
            EvalCircuitResult, EvalCircuitTrait, u384, u96, CircuitOutputsTrait, CircuitModulus, into_u96_guarantee, U96Guarantee,
            CircuitInputs, AddInputResultTrait, AddInputResult, IntoCircuitInputValue, add_circuit_input
        };
        const BN254_PRIME_LIMBS: [
            u96
            ; 4] = [
            0x6871ca8d3c208c16d87cfd47, 0xb85045b68181585d97816a91, 0x30644e72e131a029, 0x0
        ];

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

        // Add two BN254 EC points without checking if:
        // - the points are on the curve
        // - the points are not the same
        // - none of the points are the point at infinity
        fn add_ec_point_unchecked() -> (u384, u384) {
            let xP = u384 {
                limb0: 0xb3e77acb0d776ee38973b578,
                limb1: 0x7290c49d0303a7a719325387,
                limb2: 0x3104f09f1439bbd9b6e47310,
                limb3: 0x1794c7df23dbcfd21f7c96f5,
            };
            let yP = u384 {
                limb0: 0xd0ccdf6e1de037c5f25dbd53,
                limb1: 0x254a0c8d3849192e33a21665,
                limb2: 0xcc0375e474dc85925319c5ad,
                limb3: 0x59163bc09c3bb5cd5864b34,
            };
            let xQ = u384 {
                limb0: 0x42951c5be1c30dd1f90a8da3,
                limb1: 0xffa3bb5d4cc66b3c5c927fe8,
                limb2: 0xb2bef79be9fc2df478672961,
                limb3: 0x13b08e1d6ece19818bc96ea9,
            };
            let yQ = u384 {
                limb0: 0x93fd3339f961a2b9c29235bc,
                limb1: 0xf9bbad7b2c116dfe3ed68c7a,
                limb2: 0xbd2f1d7614ffe6107af3312d,
                limb3: 0x565882562afe825ad18d630,
            };
            // INPUT stack
            let (_xP, _yP, _xQ, _yQ) = (CircuitElement::<CircuitInput<0>> {}, CircuitElement::<CircuitInput<1>> {}, CircuitElement::<CircuitInput<2>> {}, CircuitElement::<CircuitInput<3>> {});

            let num = circuit_sub(_yP, _yQ);
            let den = circuit_sub(_xP, _xQ);
            let inv_den = circuit_inverse(den);
            let slope = circuit_mul(num, inv_den);
            let slope_sqr = circuit_mul(slope, slope);

            let nx = circuit_sub(circuit_sub(slope_sqr, _xP), _xQ);
            let ny = circuit_sub(circuit_mul(slope, circuit_sub(_xP, nx)), _yP);

            let modulus = TryInto::<_, CircuitModulus>::try_into(BN254_PRIME_LIMBS).unwrap(); // BN254 prime field modulus

            let mut circuit_inputs = (nx, ny,).new_inputs();
            // Fill inputs:
            circuit_inputs = circuit_inputs.next_2(xP); // in1
            circuit_inputs = circuit_inputs.next_2(yP); // in2
            circuit_inputs = circuit_inputs.next_2(xQ); // in3
            circuit_inputs = circuit_inputs.next_2(yQ); // in4

            let outputs = circuit_inputs.done_2().eval(modulus).unwrap();

            (outputs.get_output(nx), outputs.get_output(ny))
        }
    };
}

lazy_static! {
    static ref BIG_CIRCUIT: (String, Program, SierraCasmRunner) = load_cairo! {
        use core::circuit::{
            CircuitData, CircuitElement as CE, CircuitInput as CI, circuit_add, circuit_sub, circuit_mul, circuit_inverse,
            EvalCircuitResult, EvalCircuitTrait, u384, CircuitOutputsTrait, CircuitModulus,
            CircuitInputs, AddInputResultTrait, AddInputResult, IntoCircuitInputValue, add_circuit_input
        };

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

        // Taken from: https://github.com/keep-starknet-strange/garaga/blob/5c5859e6dc5515f542c310cb38a149602e774112/src/src/definitions/structs/points.cairo#L8
        // Represents a point on G1, the group of rational points on an elliptic curve over the base field.
        #[derive(Copy, Drop, Debug, PartialEq)]
        pub struct G1Point {
            pub x: u384,
            pub y: u384,
        }

        // Taken from: https://github.com/keep-starknet-strange/garaga/blob/5c5859e6dc5515f542c310cb38a149602e774112/src/src/definitions/curves.cairo#L322
        #[inline(always)]
        pub fn get_BLS12_381_modulus() -> CircuitModulus {
            let modulus = TryInto::<
                _, CircuitModulus,
            >::try_into(
                [
                    0xb153ffffb9feffffffffaaab, 0x6730d2a0f6b0f6241eabfffe, 0x434bacd764774b84f38512bf,
                    0x1a0111ea397fe69a4b1ba7b6,
                ],
            )
                .unwrap();
            modulus
        }

        // Taken from: https://github.com/keep-starknet-strange/garaga/blob/5c5859e6dc5515f542c310cb38a149602e774112/src/src/circuits/ec.cairo#L425
        // Clear cofactor of a point in the BLS12-381 elliptic curve
        #[inline(always)]
        pub fn run_CLEAR_COFACTOR_BLS12_381_circuit() -> (G1Point, G1Point, G1Point, G1Point) {
            let P = G1Point {
                x: u384 {
                    limb0: 0x23893f1bb0fdb0533584b05f,
                    limb1: 0x420d425d79dcd48b26d87814,
                    limb2: 0xc932fa90468e6b9dfd658cc9,
                    limb3: 0xe5fac70e9096e97adc6dd89,
                },
                y: u384 {
                    limb0: 0x90d1a47263d9c179e9d6bab3,
                    limb1: 0xc8f52b7ac4908e42515e61a6,
                    limb2: 0x85c60896512fc21fc50ce238,
                    limb3: 0x15bb2157a1b9aab29d66c644,
                },
            };
            let modulus = get_BLS12_381_modulus();
            // CONSTANT stack
            let in0 = CE::<CI<0>> {}; // 0x3

            // INPUT stack
            let (in1, in2) = (CE::<CI<1>> {}, CE::<CI<2>> {});
            let t0 = circuit_mul(in1, in1);
            let t1 = circuit_mul(in0, t0);
            let t2 = circuit_add(in2, in2);
            let t3 = circuit_inverse(t2);
            let t4 = circuit_mul(t1, t3);
            let t5 = circuit_mul(t4, t4);
            let t6 = circuit_sub(t5, in1);
            let t7 = circuit_sub(t6, in1);
            let t8 = circuit_sub(in1, t7);
            let t9 = circuit_mul(t4, t8);
            let t10 = circuit_sub(t9, in2);
            let t11 = circuit_sub(t10, in2);
            let t12 = circuit_sub(t7, in1);
            let t13 = circuit_inverse(t12);
            let t14 = circuit_mul(t11, t13);
            let t15 = circuit_mul(t14, t14);
            let t16 = circuit_sub(t15, t7);
            let t17 = circuit_sub(t16, in1);
            let t18 = circuit_sub(t7, t17);
            let t19 = circuit_mul(t14, t18);
            let t20 = circuit_sub(t19, t10);
            let t21 = circuit_mul(t17, t17);
            let t22 = circuit_mul(in0, t21);
            let t23 = circuit_add(t20, t20);
            let t24 = circuit_inverse(t23);
            let t25 = circuit_mul(t22, t24);
            let t26 = circuit_mul(t25, t25);
            let t27 = circuit_sub(t26, t17);
            let t28 = circuit_sub(t27, t17);
            let t29 = circuit_sub(t17, t28);
            let t30 = circuit_mul(t25, t29);
            let t31 = circuit_sub(t30, t20);
            let t32 = circuit_mul(t28, t28);
            let t33 = circuit_mul(in0, t32);
            let t34 = circuit_add(t31, t31);
            let t35 = circuit_inverse(t34);
            let t36 = circuit_mul(t33, t35);
            let t37 = circuit_mul(t36, t36);
            let t38 = circuit_sub(t37, t28);
            let t39 = circuit_sub(t38, t28);
            let t40 = circuit_sub(t28, t39);
            let t41 = circuit_mul(t36, t40);
            let t42 = circuit_sub(t41, t31);
            let t43 = circuit_sub(t42, in2);
            let t44 = circuit_sub(t39, in1);
            let t45 = circuit_inverse(t44);
            let t46 = circuit_mul(t43, t45);
            let t47 = circuit_mul(t46, t46);
            let t48 = circuit_sub(t47, t39);
            let t49 = circuit_sub(t48, in1);
            let t50 = circuit_sub(t39, t49);
            let t51 = circuit_mul(t46, t50);
            let t52 = circuit_sub(t51, t42);
            let t53 = circuit_mul(t49, t49);
            let t54 = circuit_mul(in0, t53);
            let t55 = circuit_add(t52, t52);
            let t56 = circuit_inverse(t55);
            let t57 = circuit_mul(t54, t56);
            let t58 = circuit_mul(t57, t57);
            let t59 = circuit_sub(t58, t49);
            let t60 = circuit_sub(t59, t49);
            let t61 = circuit_sub(t49, t60);
            let t62 = circuit_mul(t57, t61);
            let t63 = circuit_sub(t62, t52);
            let t64 = circuit_mul(t60, t60);
            let t65 = circuit_mul(in0, t64);
            let t66 = circuit_add(t63, t63);
            let t67 = circuit_inverse(t66);
            let t68 = circuit_mul(t65, t67);
            let t69 = circuit_mul(t68, t68);
            let t70 = circuit_sub(t69, t60);
            let t71 = circuit_sub(t70, t60);
            let t72 = circuit_sub(t60, t71);
            let t73 = circuit_mul(t68, t72);
            let t74 = circuit_sub(t73, t63);
            let t75 = circuit_mul(t71, t71);
            let t76 = circuit_mul(in0, t75);
            let t77 = circuit_add(t74, t74);
            let t78 = circuit_inverse(t77);
            let t79 = circuit_mul(t76, t78);
            let t80 = circuit_mul(t79, t79);
            let t81 = circuit_sub(t80, t71);
            let t82 = circuit_sub(t81, t71);
            let t83 = circuit_sub(t71, t82);
            let t84 = circuit_mul(t79, t83);
            let t85 = circuit_sub(t84, t74);
            let t86 = circuit_sub(t85, in2);
            let t87 = circuit_sub(t82, in1);
            let t88 = circuit_inverse(t87);
            let t89 = circuit_mul(t86, t88);
            let t90 = circuit_mul(t89, t89);
            let t91 = circuit_sub(t90, t82);
            let t92 = circuit_sub(t91, in1);
            let t93 = circuit_sub(t82, t92);
            let t94 = circuit_mul(t89, t93);
            let t95 = circuit_sub(t94, t85);
            let t96 = circuit_mul(t92, t92);
            let t97 = circuit_mul(in0, t96);
            let t98 = circuit_add(t95, t95);
            let t99 = circuit_inverse(t98);
            let t100 = circuit_mul(t97, t99);
            let t101 = circuit_mul(t100, t100);
            let t102 = circuit_sub(t101, t92);
            let t103 = circuit_sub(t102, t92);
            let t104 = circuit_sub(t92, t103);
            let t105 = circuit_mul(t100, t104);
            let t106 = circuit_sub(t105, t95);
            let t107 = circuit_mul(t103, t103);
            let t108 = circuit_mul(in0, t107);
            let t109 = circuit_add(t106, t106);
            let t110 = circuit_inverse(t109);
            let t111 = circuit_mul(t108, t110);
            let t112 = circuit_mul(t111, t111);
            let t113 = circuit_sub(t112, t103);
            let t114 = circuit_sub(t113, t103);
            let t115 = circuit_sub(t103, t114);
            let t116 = circuit_mul(t111, t115);
            let t117 = circuit_sub(t116, t106);
            let t118 = circuit_mul(t114, t114);
            let t119 = circuit_mul(in0, t118);
            let t120 = circuit_add(t117, t117);
            let t121 = circuit_inverse(t120);
            let t122 = circuit_mul(t119, t121);
            let t123 = circuit_mul(t122, t122);
            let t124 = circuit_sub(t123, t114);
            let t125 = circuit_sub(t124, t114);
            let t126 = circuit_sub(t114, t125);
            let t127 = circuit_mul(t122, t126);
            let t128 = circuit_sub(t127, t117);
            let t129 = circuit_mul(t125, t125);
            let t130 = circuit_mul(in0, t129);
            let t131 = circuit_add(t128, t128);
            let t132 = circuit_inverse(t131);
            let t133 = circuit_mul(t130, t132);
            let t134 = circuit_mul(t133, t133);
            let t135 = circuit_sub(t134, t125);
            let t136 = circuit_sub(t135, t125);
            let t137 = circuit_sub(t125, t136);
            let t138 = circuit_mul(t133, t137);
            let t139 = circuit_sub(t138, t128);
            let t140 = circuit_mul(t136, t136);
            let t141 = circuit_mul(in0, t140);
            let t142 = circuit_add(t139, t139);
            let t143 = circuit_inverse(t142);
            let t144 = circuit_mul(t141, t143);
            let t145 = circuit_mul(t144, t144);
            let t146 = circuit_sub(t145, t136);
            let t147 = circuit_sub(t146, t136);
            let t148 = circuit_sub(t136, t147);
            let t149 = circuit_mul(t144, t148);
            let t150 = circuit_sub(t149, t139);
            let t151 = circuit_mul(t147, t147);
            let t152 = circuit_mul(in0, t151);
            let t153 = circuit_add(t150, t150);
            let t154 = circuit_inverse(t153);
            let t155 = circuit_mul(t152, t154);
            let t156 = circuit_mul(t155, t155);
            let t157 = circuit_sub(t156, t147);
            let t158 = circuit_sub(t157, t147);
            let t159 = circuit_sub(t147, t158);
            let t160 = circuit_mul(t155, t159);
            let t161 = circuit_sub(t160, t150);
            let t162 = circuit_mul(t158, t158);
            let t163 = circuit_mul(in0, t162);
            let t164 = circuit_add(t161, t161);
            let t165 = circuit_inverse(t164);
            let t166 = circuit_mul(t163, t165);
            let t167 = circuit_mul(t166, t166);
            let t168 = circuit_sub(t167, t158);
            let t169 = circuit_sub(t168, t158);
            let t170 = circuit_sub(t158, t169);
            let t171 = circuit_mul(t166, t170);
            let t172 = circuit_sub(t171, t161);
            let t173 = circuit_mul(t169, t169);
            let t174 = circuit_mul(in0, t173);
            let t175 = circuit_add(t172, t172);
            let t176 = circuit_inverse(t175);
            let t177 = circuit_mul(t174, t176);
            let t178 = circuit_mul(t177, t177);
            let t179 = circuit_sub(t178, t169);
            let t180 = circuit_sub(t179, t169);
            let t181 = circuit_sub(t169, t180);
            let t182 = circuit_mul(t177, t181);
            let t183 = circuit_sub(t182, t172);
            let t184 = circuit_mul(t180, t180);
            let t185 = circuit_mul(in0, t184);
            let t186 = circuit_add(t183, t183);
            let t187 = circuit_inverse(t186);
            let t188 = circuit_mul(t185, t187);
            let t189 = circuit_mul(t188, t188);
            let t190 = circuit_sub(t189, t180);
            let t191 = circuit_sub(t190, t180);
            let t192 = circuit_sub(t180, t191);
            let t193 = circuit_mul(t188, t192);
            let t194 = circuit_sub(t193, t183);
            let t195 = circuit_sub(t194, in2);
            let t196 = circuit_sub(t191, in1);
            let t197 = circuit_inverse(t196);
            let t198 = circuit_mul(t195, t197);
            let t199 = circuit_mul(t198, t198);
            let t200 = circuit_sub(t199, t191);
            let t201 = circuit_sub(t200, in1);
            let t202 = circuit_sub(t191, t201);
            let t203 = circuit_mul(t198, t202);
            let t204 = circuit_sub(t203, t194);
            let t205 = circuit_mul(t201, t201);
            let t206 = circuit_mul(in0, t205);
            let t207 = circuit_add(t204, t204);
            let t208 = circuit_inverse(t207);
            let t209 = circuit_mul(t206, t208);
            let t210 = circuit_mul(t209, t209);
            let t211 = circuit_sub(t210, t201);
            let t212 = circuit_sub(t211, t201);
            let t213 = circuit_sub(t201, t212);
            let t214 = circuit_mul(t209, t213);
            let t215 = circuit_sub(t214, t204);
            let t216 = circuit_mul(t212, t212);
            let t217 = circuit_mul(in0, t216);
            let t218 = circuit_add(t215, t215);
            let t219 = circuit_inverse(t218);
            let t220 = circuit_mul(t217, t219);
            let t221 = circuit_mul(t220, t220);
            let t222 = circuit_sub(t221, t212);
            let t223 = circuit_sub(t222, t212);
            let t224 = circuit_sub(t212, t223);
            let t225 = circuit_mul(t220, t224);
            let t226 = circuit_sub(t225, t215);
            let t227 = circuit_mul(t223, t223);
            let t228 = circuit_mul(in0, t227);
            let t229 = circuit_add(t226, t226);
            let t230 = circuit_inverse(t229);
            let t231 = circuit_mul(t228, t230);
            let t232 = circuit_mul(t231, t231);
            let t233 = circuit_sub(t232, t223);
            let t234 = circuit_sub(t233, t223);
            let t235 = circuit_sub(t223, t234);
            let t236 = circuit_mul(t231, t235);
            let t237 = circuit_sub(t236, t226);
            let t238 = circuit_mul(t234, t234);
            let t239 = circuit_mul(in0, t238);
            let t240 = circuit_add(t237, t237);
            let t241 = circuit_inverse(t240);
            let t242 = circuit_mul(t239, t241);
            let t243 = circuit_mul(t242, t242);
            let t244 = circuit_sub(t243, t234);
            let t245 = circuit_sub(t244, t234);
            let t246 = circuit_sub(t234, t245);
            let t247 = circuit_mul(t242, t246);
            let t248 = circuit_sub(t247, t237);
            let t249 = circuit_mul(t245, t245);
            let t250 = circuit_mul(in0, t249);
            let t251 = circuit_add(t248, t248);
            let t252 = circuit_inverse(t251);
            let t253 = circuit_mul(t250, t252);
            let t254 = circuit_mul(t253, t253);
            let t255 = circuit_sub(t254, t245);
            let t256 = circuit_sub(t255, t245);
            let t257 = circuit_sub(t245, t256);
            let t258 = circuit_mul(t253, t257);
            let t259 = circuit_sub(t258, t248);
            let t260 = circuit_mul(t256, t256);
            let t261 = circuit_mul(in0, t260);
            let t262 = circuit_add(t259, t259);
            let t263 = circuit_inverse(t262);
            let t264 = circuit_mul(t261, t263);
            let t265 = circuit_mul(t264, t264);
            let t266 = circuit_sub(t265, t256);
            let t267 = circuit_sub(t266, t256);
            let t268 = circuit_sub(t256, t267);
            let t269 = circuit_mul(t264, t268);
            let t270 = circuit_sub(t269, t259);
            let t271 = circuit_mul(t267, t267);
            let t272 = circuit_mul(in0, t271);
            let t273 = circuit_add(t270, t270);
            let t274 = circuit_inverse(t273);
            let t275 = circuit_mul(t272, t274);
            let t276 = circuit_mul(t275, t275);
            let t277 = circuit_sub(t276, t267);
            let t278 = circuit_sub(t277, t267);
            let t279 = circuit_sub(t267, t278);
            let t280 = circuit_mul(t275, t279);
            let t281 = circuit_sub(t280, t270);
            let t282 = circuit_mul(t278, t278);
            let t283 = circuit_mul(in0, t282);
            let t284 = circuit_add(t281, t281);
            let t285 = circuit_inverse(t284);
            let t286 = circuit_mul(t283, t285);
            let t287 = circuit_mul(t286, t286);
            let t288 = circuit_sub(t287, t278);
            let t289 = circuit_sub(t288, t278);
            let t290 = circuit_sub(t278, t289);
            let t291 = circuit_mul(t286, t290);
            let t292 = circuit_sub(t291, t281);
            let t293 = circuit_mul(t289, t289);
            let t294 = circuit_mul(in0, t293);
            let t295 = circuit_add(t292, t292);
            let t296 = circuit_inverse(t295);
            let t297 = circuit_mul(t294, t296);
            let t298 = circuit_mul(t297, t297);
            let t299 = circuit_sub(t298, t289);
            let t300 = circuit_sub(t299, t289);
            let t301 = circuit_sub(t289, t300);
            let t302 = circuit_mul(t297, t301);
            let t303 = circuit_sub(t302, t292);
            let t304 = circuit_mul(t300, t300);
            let t305 = circuit_mul(in0, t304);
            let t306 = circuit_add(t303, t303);
            let t307 = circuit_inverse(t306);
            let t308 = circuit_mul(t305, t307);
            let t309 = circuit_mul(t308, t308);
            let t310 = circuit_sub(t309, t300);
            let t311 = circuit_sub(t310, t300);
            let t312 = circuit_sub(t300, t311);
            let t313 = circuit_mul(t308, t312);
            let t314 = circuit_sub(t313, t303);
            let t315 = circuit_mul(t311, t311);
            let t316 = circuit_mul(in0, t315);
            let t317 = circuit_add(t314, t314);
            let t318 = circuit_inverse(t317);
            let t319 = circuit_mul(t316, t318);
            let t320 = circuit_mul(t319, t319);
            let t321 = circuit_sub(t320, t311);
            let t322 = circuit_sub(t321, t311);
            let t323 = circuit_sub(t311, t322);
            let t324 = circuit_mul(t319, t323);
            let t325 = circuit_sub(t324, t314);
            let t326 = circuit_mul(t322, t322);
            let t327 = circuit_mul(in0, t326);
            let t328 = circuit_add(t325, t325);
            let t329 = circuit_inverse(t328);
            let t330 = circuit_mul(t327, t329);
            let t331 = circuit_mul(t330, t330);
            let t332 = circuit_sub(t331, t322);
            let t333 = circuit_sub(t332, t322);
            let t334 = circuit_sub(t322, t333);
            let t335 = circuit_mul(t330, t334);
            let t336 = circuit_sub(t335, t325);
            let t337 = circuit_mul(t333, t333);
            let t338 = circuit_mul(in0, t337);
            let t339 = circuit_add(t336, t336);
            let t340 = circuit_inverse(t339);
            let t341 = circuit_mul(t338, t340);
            let t342 = circuit_mul(t341, t341);
            let t343 = circuit_sub(t342, t333);
            let t344 = circuit_sub(t343, t333);
            let t345 = circuit_sub(t333, t344);
            let t346 = circuit_mul(t341, t345);
            let t347 = circuit_sub(t346, t336);
            let t348 = circuit_mul(t344, t344);
            let t349 = circuit_mul(in0, t348);
            let t350 = circuit_add(t347, t347);
            let t351 = circuit_inverse(t350);
            let t352 = circuit_mul(t349, t351);
            let t353 = circuit_mul(t352, t352);
            let t354 = circuit_sub(t353, t344);
            let t355 = circuit_sub(t354, t344);
            let t356 = circuit_sub(t344, t355);
            let t357 = circuit_mul(t352, t356);
            let t358 = circuit_sub(t357, t347);
            let t359 = circuit_mul(t355, t355);
            let t360 = circuit_mul(in0, t359);
            let t361 = circuit_add(t358, t358);
            let t362 = circuit_inverse(t361);
            let t363 = circuit_mul(t360, t362);
            let t364 = circuit_mul(t363, t363);
            let t365 = circuit_sub(t364, t355);
            let t366 = circuit_sub(t365, t355);
            let t367 = circuit_sub(t355, t366);
            let t368 = circuit_mul(t363, t367);
            let t369 = circuit_sub(t368, t358);
            let t370 = circuit_mul(t366, t366);
            let t371 = circuit_mul(in0, t370);
            let t372 = circuit_add(t369, t369);
            let t373 = circuit_inverse(t372);
            let t374 = circuit_mul(t371, t373);
            let t375 = circuit_mul(t374, t374);
            let t376 = circuit_sub(t375, t366);
            let t377 = circuit_sub(t376, t366);
            let t378 = circuit_sub(t366, t377);
            let t379 = circuit_mul(t374, t378);
            let t380 = circuit_sub(t379, t369);
            let t381 = circuit_mul(t377, t377);
            let t382 = circuit_mul(in0, t381);
            let t383 = circuit_add(t380, t380);
            let t384 = circuit_inverse(t383);
            let t385 = circuit_mul(t382, t384);
            let t386 = circuit_mul(t385, t385);
            let t387 = circuit_sub(t386, t377);
            let t388 = circuit_sub(t387, t377);
            let t389 = circuit_sub(t377, t388);
            let t390 = circuit_mul(t385, t389);
            let t391 = circuit_sub(t390, t380);
            let t392 = circuit_mul(t388, t388);
            let t393 = circuit_mul(in0, t392);
            let t394 = circuit_add(t391, t391);
            let t395 = circuit_inverse(t394);
            let t396 = circuit_mul(t393, t395);
            let t397 = circuit_mul(t396, t396);
            let t398 = circuit_sub(t397, t388);
            let t399 = circuit_sub(t398, t388);
            let t400 = circuit_sub(t388, t399);
            let t401 = circuit_mul(t396, t400);
            let t402 = circuit_sub(t401, t391);
            let t403 = circuit_mul(t399, t399);
            let t404 = circuit_mul(in0, t403);
            let t405 = circuit_add(t402, t402);
            let t406 = circuit_inverse(t405);
            let t407 = circuit_mul(t404, t406);
            let t408 = circuit_mul(t407, t407);
            let t409 = circuit_sub(t408, t399);
            let t410 = circuit_sub(t409, t399);
            let t411 = circuit_sub(t399, t410);
            let t412 = circuit_mul(t407, t411);
            let t413 = circuit_sub(t412, t402);
            let t414 = circuit_mul(t410, t410);
            let t415 = circuit_mul(in0, t414);
            let t416 = circuit_add(t413, t413);
            let t417 = circuit_inverse(t416);
            let t418 = circuit_mul(t415, t417);
            let t419 = circuit_mul(t418, t418);
            let t420 = circuit_sub(t419, t410);
            let t421 = circuit_sub(t420, t410);
            let t422 = circuit_sub(t410, t421);
            let t423 = circuit_mul(t418, t422);
            let t424 = circuit_sub(t423, t413);
            let t425 = circuit_mul(t421, t421);
            let t426 = circuit_mul(in0, t425);
            let t427 = circuit_add(t424, t424);
            let t428 = circuit_inverse(t427);
            let t429 = circuit_mul(t426, t428);
            let t430 = circuit_mul(t429, t429);
            let t431 = circuit_sub(t430, t421);
            let t432 = circuit_sub(t431, t421);
            let t433 = circuit_sub(t421, t432);
            let t434 = circuit_mul(t429, t433);
            let t435 = circuit_sub(t434, t424);
            let t436 = circuit_mul(t432, t432);
            let t437 = circuit_mul(in0, t436);
            let t438 = circuit_add(t435, t435);
            let t439 = circuit_inverse(t438);
            let t440 = circuit_mul(t437, t439);
            let t441 = circuit_mul(t440, t440);
            let t442 = circuit_sub(t441, t432);
            let t443 = circuit_sub(t442, t432);
            let t444 = circuit_sub(t432, t443);
            let t445 = circuit_mul(t440, t444);
            let t446 = circuit_sub(t445, t435);
            let t447 = circuit_mul(t443, t443);
            let t448 = circuit_mul(in0, t447);
            let t449 = circuit_add(t446, t446);
            let t450 = circuit_inverse(t449);
            let t451 = circuit_mul(t448, t450);
            let t452 = circuit_mul(t451, t451);
            let t453 = circuit_sub(t452, t443);
            let t454 = circuit_sub(t453, t443);
            let t455 = circuit_sub(t443, t454);
            let t456 = circuit_mul(t451, t455);
            let t457 = circuit_sub(t456, t446);
            let t458 = circuit_mul(t454, t454);
            let t459 = circuit_mul(in0, t458);
            let t460 = circuit_add(t457, t457);
            let t461 = circuit_inverse(t460);
            let t462 = circuit_mul(t459, t461);
            let t463 = circuit_mul(t462, t462);
            let t464 = circuit_sub(t463, t454);
            let t465 = circuit_sub(t464, t454);
            let t466 = circuit_sub(t454, t465);
            let t467 = circuit_mul(t462, t466);
            let t468 = circuit_sub(t467, t457);
            let t469 = circuit_mul(t465, t465);
            let t470 = circuit_mul(in0, t469);
            let t471 = circuit_add(t468, t468);
            let t472 = circuit_inverse(t471);
            let t473 = circuit_mul(t470, t472);
            let t474 = circuit_mul(t473, t473);
            let t475 = circuit_sub(t474, t465);
            let t476 = circuit_sub(t475, t465);
            let t477 = circuit_sub(t465, t476);
            let t478 = circuit_mul(t473, t477);
            let t479 = circuit_sub(t478, t468);
            let t480 = circuit_mul(t476, t476);
            let t481 = circuit_mul(in0, t480);
            let t482 = circuit_add(t479, t479);
            let t483 = circuit_inverse(t482);
            let t484 = circuit_mul(t481, t483);
            let t485 = circuit_mul(t484, t484);
            let t486 = circuit_sub(t485, t476);
            let t487 = circuit_sub(t486, t476);
            let t488 = circuit_sub(t476, t487);
            let t489 = circuit_mul(t484, t488);
            let t490 = circuit_sub(t489, t479);
            let t491 = circuit_mul(t487, t487);
            let t492 = circuit_mul(in0, t491);
            let t493 = circuit_add(t490, t490);
            let t494 = circuit_inverse(t493);
            let t495 = circuit_mul(t492, t494);
            let t496 = circuit_mul(t495, t495);
            let t497 = circuit_sub(t496, t487);
            let t498 = circuit_sub(t497, t487);
            let t499 = circuit_sub(t487, t498);
            let t500 = circuit_mul(t495, t499);
            let t501 = circuit_sub(t500, t490);
            let t502 = circuit_mul(t498, t498);
            let t503 = circuit_mul(in0, t502);
            let t504 = circuit_add(t501, t501);
            let t505 = circuit_inverse(t504);
            let t506 = circuit_mul(t503, t505);
            let t507 = circuit_mul(t506, t506);
            let t508 = circuit_sub(t507, t498);
            let t509 = circuit_sub(t508, t498);
            let t510 = circuit_sub(t498, t509);
            let t511 = circuit_mul(t506, t510);
            let t512 = circuit_sub(t511, t501);
            let t513 = circuit_mul(t509, t509);
            let t514 = circuit_mul(in0, t513);
            let t515 = circuit_add(t512, t512);
            let t516 = circuit_inverse(t515);
            let t517 = circuit_mul(t514, t516);
            let t518 = circuit_mul(t517, t517);
            let t519 = circuit_sub(t518, t509);
            let t520 = circuit_sub(t519, t509);
            let t521 = circuit_sub(t509, t520);
            let t522 = circuit_mul(t517, t521);
            let t523 = circuit_sub(t522, t512);
            let t524 = circuit_mul(t520, t520);
            let t525 = circuit_mul(in0, t524);
            let t526 = circuit_add(t523, t523);
            let t527 = circuit_inverse(t526);
            let t528 = circuit_mul(t525, t527);
            let t529 = circuit_mul(t528, t528);
            let t530 = circuit_sub(t529, t520);
            let t531 = circuit_sub(t530, t520);
            let t532 = circuit_sub(t520, t531);
            let t533 = circuit_mul(t528, t532);
            let t534 = circuit_sub(t533, t523);
            let t535 = circuit_mul(t531, t531);
            let t536 = circuit_mul(in0, t535);
            let t537 = circuit_add(t534, t534);
            let t538 = circuit_inverse(t537);
            let t539 = circuit_mul(t536, t538);
            let t540 = circuit_mul(t539, t539);
            let t541 = circuit_sub(t540, t531);
            let t542 = circuit_sub(t541, t531);
            let t543 = circuit_sub(t531, t542);
            let t544 = circuit_mul(t539, t543);
            let t545 = circuit_sub(t544, t534);
            let t546 = circuit_mul(t542, t542);
            let t547 = circuit_mul(in0, t546);
            let t548 = circuit_add(t545, t545);
            let t549 = circuit_inverse(t548);
            let t550 = circuit_mul(t547, t549);
            let t551 = circuit_mul(t550, t550);
            let t552 = circuit_sub(t551, t542);
            let t553 = circuit_sub(t552, t542);
            let t554 = circuit_sub(t542, t553);
            let t555 = circuit_mul(t550, t554);
            let t556 = circuit_sub(t555, t545);
            let t557 = circuit_sub(t556, in2);
            let t558 = circuit_sub(t553, in1);
            let t559 = circuit_inverse(t558);
            let t560 = circuit_mul(t557, t559);
            let t561 = circuit_mul(t560, t560);
            let t562 = circuit_sub(t561, t553);
            let t563 = circuit_sub(t562, in1);
            let t564 = circuit_sub(t553, t563);
            let t565 = circuit_mul(t560, t564);
            let t566 = circuit_sub(t565, t556);
            let t567 = circuit_mul(t563, t563);
            let t568 = circuit_mul(in0, t567);
            let t569 = circuit_add(t566, t566);
            let t570 = circuit_inverse(t569);
            let t571 = circuit_mul(t568, t570);
            let t572 = circuit_mul(t571, t571);
            let t573 = circuit_sub(t572, t563);
            let t574 = circuit_sub(t573, t563);
            let t575 = circuit_sub(t563, t574);
            let t576 = circuit_mul(t571, t575);
            let t577 = circuit_sub(t576, t566);
            let t578 = circuit_mul(t574, t574);
            let t579 = circuit_mul(in0, t578);
            let t580 = circuit_add(t577, t577);
            let t581 = circuit_inverse(t580);
            let t582 = circuit_mul(t579, t581);
            let t583 = circuit_mul(t582, t582);
            let t584 = circuit_sub(t583, t574);
            let t585 = circuit_sub(t584, t574);
            let t586 = circuit_sub(t574, t585);
            let t587 = circuit_mul(t582, t586);
            let t588 = circuit_sub(t587, t577);
            let t589 = circuit_mul(t585, t585);
            let t590 = circuit_mul(in0, t589);
            let t591 = circuit_add(t588, t588);
            let t592 = circuit_inverse(t591);
            let t593 = circuit_mul(t590, t592);
            let t594 = circuit_mul(t593, t593);
            let t595 = circuit_sub(t594, t585);
            let t596 = circuit_sub(t595, t585);
            let t597 = circuit_sub(t585, t596);
            let t598 = circuit_mul(t593, t597);
            let t599 = circuit_sub(t598, t588);
            let t600 = circuit_mul(t596, t596);
            let t601 = circuit_mul(in0, t600);
            let t602 = circuit_add(t599, t599);
            let t603 = circuit_inverse(t602);
            let t604 = circuit_mul(t601, t603);
            let t605 = circuit_mul(t604, t604);
            let t606 = circuit_sub(t605, t596);
            let t607 = circuit_sub(t606, t596);
            let t608 = circuit_sub(t596, t607);
            let t609 = circuit_mul(t604, t608);
            let t610 = circuit_sub(t609, t599);
            let t611 = circuit_mul(t607, t607);
            let t612 = circuit_mul(in0, t611);
            let t613 = circuit_add(t610, t610);
            let t614 = circuit_inverse(t613);
            let t615 = circuit_mul(t612, t614);
            let t616 = circuit_mul(t615, t615);
            let t617 = circuit_sub(t616, t607);
            let t618 = circuit_sub(t617, t607);
            let t619 = circuit_sub(t607, t618);
            let t620 = circuit_mul(t615, t619);
            let t621 = circuit_sub(t620, t610);
            let t622 = circuit_mul(t618, t618);
            let t623 = circuit_mul(in0, t622);
            let t624 = circuit_add(t621, t621);
            let t625 = circuit_inverse(t624);
            let t626 = circuit_mul(t623, t625);
            let t627 = circuit_mul(t626, t626);
            let t628 = circuit_sub(t627, t618);
            let t629 = circuit_sub(t628, t618);
            let t630 = circuit_sub(t618, t629);
            let t631 = circuit_mul(t626, t630);
            let t632 = circuit_sub(t631, t621);
            let t633 = circuit_mul(t629, t629);
            let t634 = circuit_mul(in0, t633);
            let t635 = circuit_add(t632, t632);
            let t636 = circuit_inverse(t635);
            let t637 = circuit_mul(t634, t636);
            let t638 = circuit_mul(t637, t637);
            let t639 = circuit_sub(t638, t629);
            let t640 = circuit_sub(t639, t629);
            let t641 = circuit_sub(t629, t640);
            let t642 = circuit_mul(t637, t641);
            let t643 = circuit_sub(t642, t632);
            let t644 = circuit_mul(t640, t640);
            let t645 = circuit_mul(in0, t644);
            let t646 = circuit_add(t643, t643);
            let t647 = circuit_inverse(t646);
            let t648 = circuit_mul(t645, t647);
            let t649 = circuit_mul(t648, t648);
            let t650 = circuit_sub(t649, t640);
            let t651 = circuit_sub(t650, t640);
            let t652 = circuit_sub(t640, t651);
            let t653 = circuit_mul(t648, t652);
            let t654 = circuit_sub(t653, t643);
            let t655 = circuit_mul(t651, t651);
            let t656 = circuit_mul(in0, t655);
            let t657 = circuit_add(t654, t654);
            let t658 = circuit_inverse(t657);
            let t659 = circuit_mul(t656, t658);
            let t660 = circuit_mul(t659, t659);
            let t661 = circuit_sub(t660, t651);
            let t662 = circuit_sub(t661, t651);
            let t663 = circuit_sub(t651, t662);
            let t664 = circuit_mul(t659, t663);
            let t665 = circuit_sub(t664, t654);
            let t666 = circuit_mul(t662, t662);
            let t667 = circuit_mul(in0, t666);
            let t668 = circuit_add(t665, t665);
            let t669 = circuit_inverse(t668);
            let t670 = circuit_mul(t667, t669);
            let t671 = circuit_mul(t670, t670);
            let t672 = circuit_sub(t671, t662);
            let t673 = circuit_sub(t672, t662);
            let t674 = circuit_sub(t662, t673);
            let t675 = circuit_mul(t670, t674);
            let t676 = circuit_sub(t675, t665);
            let t677 = circuit_mul(t673, t673);
            let t678 = circuit_mul(in0, t677);
            let t679 = circuit_add(t676, t676);
            let t680 = circuit_inverse(t679);
            let t681 = circuit_mul(t678, t680);
            let t682 = circuit_mul(t681, t681);
            let t683 = circuit_sub(t682, t673);
            let t684 = circuit_sub(t683, t673);
            let t685 = circuit_sub(t673, t684);
            let t686 = circuit_mul(t681, t685);
            let t687 = circuit_sub(t686, t676);
            let t688 = circuit_mul(t684, t684);
            let t689 = circuit_mul(in0, t688);
            let t690 = circuit_add(t687, t687);
            let t691 = circuit_inverse(t690);
            let t692 = circuit_mul(t689, t691);
            let t693 = circuit_mul(t692, t692);
            let t694 = circuit_sub(t693, t684);
            let t695 = circuit_sub(t694, t684);
            let t696 = circuit_sub(t684, t695);
            let t697 = circuit_mul(t692, t696);
            let t698 = circuit_sub(t697, t687);
            let t699 = circuit_mul(t695, t695);
            let t700 = circuit_mul(in0, t699);
            let t701 = circuit_add(t698, t698);
            let t702 = circuit_inverse(t701);
            let t703 = circuit_mul(t700, t702);
            let t704 = circuit_mul(t703, t703);
            let t705 = circuit_sub(t704, t695);
            let t706 = circuit_sub(t705, t695);
            let t707 = circuit_sub(t695, t706);
            let t708 = circuit_mul(t703, t707);
            let t709 = circuit_sub(t708, t698);
            let t710 = circuit_mul(t706, t706);
            let t711 = circuit_mul(in0, t710);
            let t712 = circuit_add(t709, t709);
            let t713 = circuit_inverse(t712);
            let t714 = circuit_mul(t711, t713);
            let t715 = circuit_mul(t714, t714);
            let t716 = circuit_sub(t715, t706);
            let t717 = circuit_sub(t716, t706);
            let t718 = circuit_sub(t706, t717);
            let t719 = circuit_mul(t714, t718);
            let t720 = circuit_sub(t719, t709);
            let t721 = circuit_mul(t717, t717);
            let t722 = circuit_mul(in0, t721);
            let t723 = circuit_add(t720, t720);
            let t724 = circuit_inverse(t723);
            let t725 = circuit_mul(t722, t724);
            let t726 = circuit_mul(t725, t725);
            let t727 = circuit_sub(t726, t717);
            let t728 = circuit_sub(t727, t717);
            let t729 = circuit_sub(t717, t728);
            let t730 = circuit_mul(t725, t729);
            let t731 = circuit_sub(t730, t720);
            let t732 = circuit_mul(t728, t728);
            let t733 = circuit_mul(in0, t732);
            let t734 = circuit_add(t731, t731);
            let t735 = circuit_inverse(t734);
            let t736 = circuit_mul(t733, t735);
            let t737 = circuit_mul(t736, t736);
            let t738 = circuit_sub(t737, t728);
            let t739 = circuit_sub(t738, t728);
            let t740 = circuit_sub(t728, t739);
            let t741 = circuit_mul(t736, t740);
            let t742 = circuit_sub(t741, t731);
            let t743 = circuit_sub(in2, t742);
            let t744 = circuit_sub(in1, t739);
            let t745 = circuit_inverse(t744);
            let t746 = circuit_mul(t743, t745);
            let t747 = circuit_mul(t746, t746);
            let t748 = circuit_sub(t747, in1);
            let t749 = circuit_sub(t748, t739);
            let t750 = circuit_sub(in1, t749);
            let t751 = circuit_mul(t746, t750);
            let t752 = circuit_sub(t751, in2);

            let modulus = modulus;

            let mut circuit_inputs = (t749, t752).new_inputs();
            // Prefill constants:
            circuit_inputs = circuit_inputs.next_2([0x3, 0x0, 0x0, 0x0]); // in0
            // Fill inputs:
            circuit_inputs = circuit_inputs.next_2(P.x); // in1
            circuit_inputs = circuit_inputs.next_2(P.y); // in2

            let outputs = circuit_inputs.done_2().eval(modulus).unwrap();
            let s = outputs.get_output(t240);
            let r = outputs.get_output(t300);
            let t = outputs.get_output(t190);
            let q = outputs.get_output(t10);
            let v = outputs.get_output(t45);
            let u = outputs.get_output(t650);
            let res2 = G1Point {x: s, y: r};
            let res3 = G1Point {x: t, y: q};
            let res4 = G1Point {x:v, y:u};
            let res: G1Point = G1Point { x: outputs.get_output(t749), y: outputs.get_output(t752) };
            return (res,res2, res3, res4);
        }
    };
}

#[test]
fn test_circuit_guarantee_first_limb() {
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
fn test_circuit_guarantee_last_limb() {
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
fn test_circuit_guarantee_middle_limb() {
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
fn test_circuit_add() {
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
fn test_circuit_sub() {
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
fn test_circuit_mul() {
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
fn test_circuit_inv() {
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
fn test_circuit_full() {
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
fn test_circuit_fail() {
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
fn test_circuit_into_u96_guarantee() {
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
fn test_circuit_y_inv_x_neg_over_y_bn254() {
    let program = &GARAGA_CIRCUITS;

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
fn test_circuit_batch_3_mod_bn254() {
    let program = &GARAGA_CIRCUITS;

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
    let program = &GARAGA_CIRCUITS;

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

#[test]
fn test_circuit_clear_cofactor_bls12_381() {
    let program = &BIG_CIRCUIT;

    let result_vm = run_vm_program(
        program,
        "run_CLEAR_COFACTOR_BLS12_381_circuit",
        vec![],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();

    let result_native = run_native_program(
        program,
        "run_CLEAR_COFACTOR_BLS12_381_circuit",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program
            .2
            .find_function("run_CLEAR_COFACTOR_BLS12_381_circuit")
            .unwrap()
            .id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}

#[test]
fn test_circuit_add_ec_point_unchecked() {
    let program = &KAKAROT_CIRCUIT;

    let result_vm = run_vm_program(
        program,
        "add_ec_point_unchecked",
        vec![],
        Some(DEFAULT_GAS as usize),
    )
    .unwrap();

    let result_native = run_native_program(
        program,
        "add_ec_point_unchecked",
        &[],
        Some(DEFAULT_GAS),
        Option::<DummySyscallHandler>::None,
    );

    compare_outputs(
        &program.1,
        &program
            .2
            .find_function("add_ec_point_unchecked")
            .unwrap()
            .id,
        &result_vm,
        &result_native,
    )
    .unwrap();
}
