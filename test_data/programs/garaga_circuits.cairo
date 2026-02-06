/// https://github.com/keep-starknet-strange/garaga
/// 
/// MIT License
/// 
/// Copyright (c) 2023 Keep StarkNet Strange
/// 
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to deal
/// in the Software without restriction, including without limitation the rights
/// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
/// copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
/// 
/// The above copyright notice and this permission notice shall be included in all
/// copies or substantial portions of the Software.
/// 
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
/// SOFTWARE.

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
