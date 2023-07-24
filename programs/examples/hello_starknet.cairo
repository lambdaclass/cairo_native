use core::{debug::PrintTrait, starknet::get_block_hash_syscall};

fn main() {
    match get_block_hash_syscall(0_u64) {
        Result::Ok(x) => x.print(),
        Result::Err(e) => {
            'Syscall returned an error:'.print();
            '  get_block_hash'.print();
        },
    }
}
