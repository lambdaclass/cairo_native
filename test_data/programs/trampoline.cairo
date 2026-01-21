fn main(
    b0: u64,            // Pedersen
    b1: u64,            // RangeCheck
    b2: u64,            // Bitwise
    b3: u128,           // GasBuiltin
    b4: u64,            // System
    arg0: Span<felt252> // Arguments
) -> (u64, u64, u64, u128, u64, Span<felt252>) {
    (b0, b1, b2, b3, b4, arg0)
}
