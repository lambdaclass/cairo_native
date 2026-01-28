use starknet::storage_access::{StorageAddress, storage_address_to_felt252};

fn run_program(value: StorageAddress) -> felt252 {
    storage_address_to_felt252(value)
}
