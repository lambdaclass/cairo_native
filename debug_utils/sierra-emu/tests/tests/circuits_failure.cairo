use core::circuit::{
    RangeCheck96, AddMod, MulMod, u96, CircuitElement, CircuitInput, circuit_add, circuit_sub,
    circuit_mul, circuit_inverse, EvalCircuitTrait, u384, CircuitOutputsTrait, CircuitModulus,
    AddInputResultTrait, CircuitInputs,
};

fn main() {
    let in1 = CircuitElement::<CircuitInput<0>> {};
    let inv = circuit_inverse(in1);

    let modulus = TryInto::<_, CircuitModulus>::try_into([7, 0, 0, 0]).unwrap();
    let outputs = (inv,)
        .new_inputs()
        .next([0, 0, 0, 0])
        .done()
        .eval(modulus)
        .unwrap();
    
    outputs.get_output(inv);
}
