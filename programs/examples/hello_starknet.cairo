use core::clone::Clone;
use core::{
    array::ArrayTrait, debug::PrintTrait, option::OptionTrait,
    starknet::{
        call_contract_syscall, contract_address_try_from_felt252, emit_event_syscall,
        get_block_hash_syscall, storage_address_try_from_felt252, storage_read_syscall,
        storage_write_syscall,
    }
};

fn main() {
    match call_contract_syscall(
        contract_address_try_from_felt252(1234).unwrap(),
        5678,
        {
            let mut data = ArrayTrait::<felt252>::new();
            data.append(1234);
            data.append(5678);
            data.span()
        }
    ) {
        Result::Ok(x) => x.snapshot.clone().print(),
        Result::Err(e) => {
            'Syscall returned an error:'.print();
            '  call_contract'.print();
        }
    }

    match emit_event_syscall(
        {
            let mut data = ArrayTrait::<felt252>::new();
            data.append(1234);
            data.append(5678);
            data.span()
        },
        {
            let mut data = ArrayTrait::<felt252>::new();
            data.append(8765);
            data.append(4321);
            data.span()
        }
    ) {
        Result::Ok(_) => {},
        Result::Err(e) => {
            'Syscall returned an error:'.print();
            '  emit_event'.print();
        }
    }

    match get_block_hash_syscall(0_u64) {
        Result::Ok(x) => x.print(),
        Result::Err(e) => {
            'Syscall returned an error:'.print();
            '  get_block_hash'.print();
        },
    }

    match storage_read_syscall(0, storage_address_try_from_felt252(1234).unwrap()) {
        Result::Ok(x) => x.print(),
        Result::Err(e) => {
            'Syscall returned an error:'.print();
            '  storage_read'.print();
        }
    }

    match storage_write_syscall(0, storage_address_try_from_felt252(1234).unwrap(), 2345) {
        Result::Ok(_) => {},
        Result::Err(e) => {
            'Syscall returned an error:'.print();
            '  storage_write'.print();
        }
    }
}
