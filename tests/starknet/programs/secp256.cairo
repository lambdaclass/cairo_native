use core::starknet::secp256k1::{
    Secp256k1Point, secp256k1_new_syscall, secp256k1_add_syscall, secp256k1_mul_syscall,
    secp256k1_get_point_from_x_syscall, secp256k1_get_xy_syscall,
};

fn secp256k1_new(x: u256, y: u256) -> Result<Option<Secp256k1Point>, Array<felt252>> {
    secp256k1_new_syscall(x, y)
}

fn secp256k1_add(p0: Secp256k1Point, p1: Secp256k1Point) -> Result<Secp256k1Point, Array<felt252>> {
    secp256k1_add_syscall(p0, p1)
}

fn secp256k1_mul(p: Secp256k1Point, scalar: u256) -> Result<Secp256k1Point, Array<felt252>> {
    secp256k1_mul_syscall(p, scalar)
}

fn secp256k1_get_point_from_x(
    p: u256, y_parity: bool
) -> Result<Option<Secp256k1Point>, Array<felt252>> {
    secp256k1_get_point_from_x_syscall(p, y_parity)
}

fn secp256k1_get_xy(p: Secp256k1Point) -> Result<(u256, u256), Array<felt252>> {
    secp256k1_get_xy_syscall(p)
}
