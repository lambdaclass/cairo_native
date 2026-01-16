/// https://github.com/kkrt-labs/kakarot
/// 
/// MIT License
/// 
/// Copyright (c) 2022 Abdel @ StarkWare
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
