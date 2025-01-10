 use core::poseidon::hades_permutation;
 fn run_test(a: felt252, b: felt252, c: felt252) -> (felt252, felt252, felt252) {
     hades_permutation(a, b, c)
 }
