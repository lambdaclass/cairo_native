// This file is completely inspired in LambdaClass' cairo-vm blake compression implementation
// For more info check: https://github.com/lambdaclass/cairo-vm/blob/main/vm/src/hint_processor/builtin_hint_processor/blake2s_hash.rs#L24

use std::ops::Shl;

pub const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

const SIGMA: [[usize; 16]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];

fn right_rot(value: u32, n: u32) -> u32 {
    (value >> n) | ((value & (1_u32.shl(n) - 1)) << (32 - n))
}

fn mix(a: u32, b: u32, c: u32, d: u32, m0: u32, m1: u32) -> (u32, u32, u32, u32) {
    let a = a.wrapping_add(b).wrapping_add(m0);
    let d = right_rot(d ^ a, 16);
    let c = c.wrapping_add(d);
    let b = right_rot(b ^ c, 12);
    let a = a.wrapping_add(b).wrapping_add(m1);
    let d = right_rot(d ^ a, 8);
    let c = c.wrapping_add(d);
    let b = right_rot(b ^ c, 7);
    (a, b, c, d)
}

fn blake_round(mut state: Vec<u32>, message: &[u32; 16], sigma: [usize; 16]) -> Vec<u32> {
    (state[0], state[4], state[8], state[12]) = mix(
        state[0],
        state[4],
        state[8],
        state[12],
        message[sigma[0]],
        message[sigma[1]],
    );
    (state[1], state[5], state[9], state[13]) = mix(
        state[1],
        state[5],
        state[9],
        state[13],
        message[sigma[2]],
        message[sigma[3]],
    );
    (state[2], state[6], state[10], state[14]) = mix(
        state[2],
        state[6],
        state[10],
        state[14],
        message[sigma[4]],
        message[sigma[5]],
    );
    (state[3], state[7], state[11], state[15]) = mix(
        state[3],
        state[7],
        state[11],
        state[15],
        message[sigma[6]],
        message[sigma[7]],
    );
    (state[0], state[5], state[10], state[15]) = mix(
        state[0],
        state[5],
        state[10],
        state[15],
        message[sigma[8]],
        message[sigma[9]],
    );
    (state[1], state[6], state[11], state[12]) = mix(
        state[1],
        state[6],
        state[11],
        state[12],
        message[sigma[10]],
        message[sigma[11]],
    );
    (state[2], state[7], state[8], state[13]) = mix(
        state[2],
        state[7],
        state[8],
        state[13],
        message[sigma[12]],
        message[sigma[13]],
    );
    (state[3], state[4], state[9], state[14]) = mix(
        state[3],
        state[4],
        state[9],
        state[14],
        message[sigma[14]],
        message[sigma[15]],
    );
    state
}

pub fn blake2s_compress(
    h: &[u32; 8],
    message: &[u32; 16],
    t0: u32,
    t1: u32,
    f0: u32,
    f1: u32,
) -> [u32; 8] {
    let mut state = h.to_vec();
    state.extend(&IV[0..4]);
    state.extend(&vec![
        (IV[4] ^ t0),
        (IV[5] ^ t1),
        (IV[6] ^ f0),
        (IV[7] ^ f1),
    ]);
    for sigma_list in SIGMA {
        state = blake_round(state, message, sigma_list);
    }
    let mut new_state = [0; 8];
    for i in 0..8 {
        new_state[i] = h[i] ^ state[i] ^ state[8 + i];
    }
    new_state
}
