#[starknet::interface]
trait IKeccak<TContractState> {
    fn cairo_keccak_test(self: @TContractState) -> felt252;
}

#[starknet::contract]
mod Keccak {
    use core::clone::Clone;
    use array::{Array, ArrayTrait};
    use core::traits::Into;

    #[storage]
    struct Storage {}

    #[abi(embed_v0)]
    impl Keccak of super::IKeccak<ContractState> {
        fn cairo_keccak_test(self: @ContractState) -> felt252 {
            let mut input = array![
                0x0000000000000001,
                0x0000000000000002,
                0x0000000000000003,
                0x0000000000000004,
                0x0000000000000005,
                0x0000000000000006,
                0x0000000000000007,
                0x0000000000000008,
                0x0000000000000009,
                0x000000000000000a,
                0x000000000000000b,
                0x000000000000000c,
                0x000000000000000d
            ];

            // We must clone the array to be used in the second part, as it's modified by `cairo_keccak`.
            let mut _orig_array = input.clone();

            let _res = keccak::cairo_keccak(ref input, 0x11000010, 4);

            1
        }
    }
}
