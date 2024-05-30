use core::starknet::{
    call_contract_syscall, class_hash_const, contract_address_const, ContractAddress,
    deploy_syscall, emit_event_syscall, ExecutionInfo, get_block_hash_syscall,
    keccak_syscall,
    library_call_syscall, replace_class_syscall, send_message_to_l1_syscall,
    storage_address_try_from_felt252, storage_read_syscall, storage_write_syscall, SyscallResult,
    testing::cheatcode,
};
use core::starknet::syscalls::get_execution_info_syscall;
use core::starknet::syscalls::get_execution_info_v2_syscall;

fn get_block_hash() -> SyscallResult<felt252> {
    get_block_hash_syscall(0)
}

fn get_execution_info() -> SyscallResult<Box<core::starknet::info::ExecutionInfo>> {
     get_execution_info_syscall()
}

fn get_execution_info_v2() -> SyscallResult<Box<core::starknet::info::v2::ExecutionInfo>> {
    get_execution_info_v2_syscall()
}

fn deploy() -> SyscallResult<(ContractAddress, Span<felt252>)> {
    deploy_syscall(class_hash_const::<0>(), 0, array![].span(), false)
}

fn replace_class() -> SyscallResult<()> {
    replace_class_syscall(class_hash_const::<0>())
}

fn library_call() -> SyscallResult<Span<felt252>> {
    library_call_syscall(class_hash_const::<0>(), 0, array![].span())
}

fn call_contract() -> SyscallResult<Span<felt252>> {
    call_contract_syscall(contract_address_const::<0>(), 0, array![].span())
}

fn storage_read() -> felt252 {
    storage_read_syscall(0, storage_address_try_from_felt252(0).unwrap()).unwrap()
}

fn storage_write() {
    storage_write_syscall(0, storage_address_try_from_felt252(0).unwrap(), 0).unwrap()
}

fn emit_event() -> SyscallResult<()> {
    emit_event_syscall(array![].span(), array![].span())
}

fn send_message_to_l1() -> SyscallResult<()> {
    send_message_to_l1_syscall(3, array![2].span())
}

fn keccak() -> SyscallResult<u256> {
    keccak_syscall(array![].span())
}

fn set_sequencer_address() -> Span<felt252> {
    let address = 123;
    return cheatcode::<'set_sequencer_address'>(array![address].span());
}
