#[starknet::interface]
trait IKeccak<TContractState> {
    fn cairo_keccak_test(self: @TContractState) -> felt252;
}

#[starknet::contract]
mod Keccak {
    use core::clone::Clone;
    use array::{Array, ArrayTrait};
    use core::traits::Into;
    use core::debug::PrintTrait;

    #[storage]
    struct Storage {}

    #[abi(embed_v0)]
    impl Keccak of super::IKeccak<ContractState> {
        fn cairo_keccak_test(self: @ContractState) -> felt252 {
            let input : Array::<u64> = array![1,2,4,5,6,6,7,2,3,4,4,5,5,6,7,7,2];
            let output = starknet::syscalls::keccak_syscall(input.span()).unwrap();

            if output.low == 0x9293867273ef341e81577655f28aeade && output.high == 0xf70cba9bb86caa97b086fdfa3df602ed {
                panic_with_felt252('arguments swapped');
            }
            output.low.into()
        }
    }
}
