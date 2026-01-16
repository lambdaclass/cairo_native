use traits::TryInto;

// u128 doesn't have this functionality yet
fn main() -> (Option<u8>, Option<u8>,
              Option<u16>, Option<u16>,
              Option<u32>, Option<u32>,
              Option<u64>, Option<u64>,
              // Option<u128>, Option<u128>,
              ) {

   (
    255.try_into(), 256.try_into(),
    65535.try_into(), 65536.try_into(),
    4294967295.try_into(), 4294967296.try_into(),
    18446744073709551615.try_into(), 18446744073709551616.try_into(),
    // 340282366920938463463374607431768211455.try_into(), 340282366920938463463374607431768211456.try_into(),
   )
}
