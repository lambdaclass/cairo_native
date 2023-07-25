use core::clone::Clone;
use core::{
    array::ArrayTrait, debug::PrintTrait, option::OptionTrait,
    starknet::{call_contract_syscall, contract_address_try_from_felt252, get_block_hash_syscall}
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

    match get_block_hash_syscall(0_u64) {
        Result::Ok(x) => x.print(),
        Result::Err(e) => {
            'Syscall returned an error:'.print();
            '  get_block_hash'.print();
        },
    }
}
