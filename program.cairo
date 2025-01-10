use core::integer::upcast;
fn run_test(
    v8: u8, v16: u16, v32: u32, v64: u64, v128: u128, v248: bytes31
) -> (
    (u8,),
    (u16, u16),
    (u32, u32, u32),
    (u64, u64, u64, u64),
    (u128, u128, u128, u128, u128),
    (bytes31, bytes31, bytes31, bytes31, bytes31, bytes31)
) {
    (
        (upcast(v8),),
        (upcast(v8), upcast(v16)),
        (upcast(v8), upcast(v16), upcast(v32)),
        (upcast(v8), upcast(v16), upcast(v32), upcast(v64)),
        (upcast(v8), upcast(v16), upcast(v32), upcast(v64), upcast(v128)),
        (upcast(v8), upcast(v16), upcast(v32), upcast(v64), upcast(v128), upcast(v248)),
    )
}
