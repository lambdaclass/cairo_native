use core::circuit::{
    AddInputResultTrait, AddMod, CircuitElement, CircuitInput, CircuitInputs,
    CircuitModulus, CircuitOutputsTrait, EvalCircuitTrait, MulMod, RangeCheck96,
    circuit_add, circuit_inverse, circuit_mul, circuit_sub, u384, u96,
};
use core::num::traits::Zero;
use core::traits::TryInto;

fn main() {
    let _in0 = CircuitElement::<CircuitInput<0>> {};
    let _out0 = circuit_inverse(_in0);

    let _modulus = TryInto::<_, CircuitModulus>::try_into([55, 0, 0, 0]).unwrap();
    (_out0,)
        .new_inputs()
        .next([11, 0, 0, 0])
        .done()
        .eval(_modulus)
        .unwrap_err();
    (_out0,)
        .new_inputs()
        .next([11, 0, 0, 0])
        .done()
        .eval(_modulus)
        .unwrap_err();
}
