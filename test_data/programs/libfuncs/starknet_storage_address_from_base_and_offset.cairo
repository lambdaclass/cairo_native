use starknet::storage_access::{StorageAddress, StorageBaseAddress, storage_address_from_base_and_offset};

fn run_program(addr: StorageBaseAddress, offset: u8) -> StorageAddress {
    storage_address_from_base_and_offset(addr, offset)
}
