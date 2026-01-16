use core::circuit::{
    RangeCheck96, AddMod, MulMod, u96, CircuitElement, CircuitInput, circuit_add,
    circuit_sub, circuit_mul, circuit_inverse, EvalCircuitTrait, u384,
    CircuitOutputsTrait, CircuitModulus, AddInputResultTrait, CircuitInputs,
};

fn main() -> u384 {
    let in1 = CircuitElement::<CircuitInput<0>> {};
    let in2 = CircuitElement::<CircuitInput<1>> {};
    let mul = circuit_mul(in1, in2);

    let modulus = TryInto::<_, CircuitModulus>::try_into([
        0xffffffffffffffffffffffff,
        0xffffffffffffffffffffffff,
        0xffffffffffffffffffffffff,
        0xffffffffffffffffffffffff,
    ])
    .unwrap();

    let outputs = (mul,)
        .new_inputs()
        .next([0, 0, 0, 0xffffffffffffffffffffffff])
        .next([16, 0, 0, 0])
        .done()
        .eval(modulus)
        .unwrap();

    outputs.get_output(mul)
}