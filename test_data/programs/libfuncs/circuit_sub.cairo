use core::circuit::{
    RangeCheck96, AddMod, MulMod, u96, CircuitElement, CircuitInput, circuit_add,
    circuit_sub, circuit_mul, circuit_inverse, EvalCircuitTrait, u384,
    CircuitOutputsTrait, CircuitModulus, AddInputResultTrait, CircuitInputs,
};

fn main() -> u384 {
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
