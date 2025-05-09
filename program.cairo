use core::circuit::{
    AddInputResultTrait, AddMod, CircuitElement, CircuitInput, CircuitInputs, CircuitModulus,
    CircuitOutputsTrait, EvalCircuitTrait, MulMod, RangeCheck96, circuit_add, circuit_inverse,
    circuit_mul, circuit_sub, u384, u96,
};
use core::num::traits::Zero;
use core::traits::TryInto;

fn test_into_u384() -> u384 {
    0x100000023000000450000006700000089000000ab000000cd000000ef0000000_u256
        .into()
}
