use core::blake::{blake2s_compress, blake2s_finalize};

fn run_test() -> [u32; 8] nopanic {
    let initial_state: Box<[u32; 8]> = BoxTrait::new([
        0x6B08E647, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
    ]);
    // This number represents the bytes for "abc" string.
    let abc_bytes = 0x00636261;
    let msg: Box<[u32; 16]>  = BoxTrait::new([abc_bytes, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

    blake2s_finalize(initial_state, 3, msg).unbox()
}
