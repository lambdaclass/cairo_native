use integer::upcast;
use integer::downcast;

fn main() -> (Option<u8>,
              Option<u8>, Option<u16>,
              Option<u8>, Option<u16>, Option<u32>,
              Option<u8>, Option<u16>, Option<u32>, Option<u64>,
              Option<u8>, Option<u16>, Option<u32>, Option<u64>, Option<u128>) {
    let x8 = 2_u8;
    let x16 = 4_u16;
    let x32 = 1024_u32;
    let x64 = 8_u64;
    let x128 = 10_u128;

    let downcast_x8_to_8: Option<u8> = downcast(x8);

    let downcast_x16_to_8: Option<u8> = downcast(x16);
    let downcast_x16_to_16: Option<u16> = downcast(x16);

    let downcast_x32_to_8: Option<u8> = downcast(x32);
    let downcast_x32_to_16: Option<u16> = downcast(x32);
    let downcast_x32_to_32: Option<u32> = downcast(x32);

    let downcast_x64_to_8: Option<u8> = downcast(x64);
    let downcast_x64_to_16: Option<u16> = downcast(x64);
    let downcast_x64_to_32: Option<u32> = downcast(x64);
    let downcast_x64_to_64: Option<u64> = downcast(x64);

    let x128_to_8: Option<u8> = downcast(x128);
    let x128_to_16: Option<u16> = downcast(x128);
    let x128_to_32: Option<u32> = downcast(x128);
    let x128_to_64: Option<u64> = downcast(x128);
    let x128_to_128: Option<u128> = downcast(x128);

    (downcast_x8_to_8, downcast_x16_to_8, downcast_x16_to_16,
    downcast_x32_to_8, downcast_x32_to_16, downcast_x32_to_32,
    downcast_x64_to_8, downcast_x64_to_16, downcast_x64_to_32, downcast_x64_to_64,
    x128_to_8, x128_to_16, x128_to_32, x128_to_64, x128_to_128)
}
