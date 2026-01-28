use starknet::{get_block_hash_syscall, SyscallResultTrait};

fn run_test() -> felt252 {
    42
}

fn get_block_hash() -> felt252 {
    get_block_hash_syscall(1).unwrap_syscall()
}
