use debug::PrintTrait;
//use poseidon::hades_permutation;

// FIXME: Hack to make it compile.
extern fn hades_permutation(
    s0: felt252, s1: felt252, s2: felt252
) -> (felt252, felt252, felt252) implicits(Poseidon) nopanic;

fn test_pedersen() -> (felt252, felt252, felt252) {
    hades_permutation(1, 2, 3)
}

fn main() {
    let (r0, r1, r2) = test_pedersen();
    r0.print();
    r1.print();
    r2.print();
}
