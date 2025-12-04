#[starknet::interface]
            trait ISimpleStorage<TContractState> {
                fn get(self: @TContractState) -> u128;
            }

            #[starknet::contract]
            mod contract {
                #[storage]
                struct Storage {}

                #[abi(embed_v0)]
                impl ISimpleStorageImpl of super::ISimpleStorage<ContractState> {
                    fn get(self: @ContractState) -> u128 {
                        42
                    }
                }
            }
