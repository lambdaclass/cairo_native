use starknet::storage_access::{StorageAddress, storage_address_try_from_felt252};

fn run_program(value: felt252) -> Option<StorageAddress> {
    storage_address_try_from_felt252(value)
}
