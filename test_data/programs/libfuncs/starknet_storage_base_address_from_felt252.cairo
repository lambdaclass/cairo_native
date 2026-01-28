use starknet::storage_access::{StorageBaseAddress, storage_base_address_from_felt252};

fn run_program(value: felt252) -> StorageBaseAddress {
    storage_base_address_from_felt252(value)
}
