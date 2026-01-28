use starknet::storage_access::{StorageAddress, StorageBaseAddress, storage_address_from_base};

fn run_program(value: StorageBaseAddress) -> StorageAddress {
    storage_address_from_base(value)
}
