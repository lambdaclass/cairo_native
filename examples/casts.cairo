use integer::downcast;
use integer::upcast;

fn main() {
    let x8 = 0_u8;
    let x16 = 0_u16;
    let x32 = 0_u32;
    let x64 = 0_u64;
    let x128 = 0_u128;

    // Upcasts.
    let upcast_x8_to_8: u8 = upcast(x8);
    let upcast_x8_to_16: u16 = upcast(x8);
    let upcast_x8_to_32: u32 = upcast(x8);
    let upcast_x8_to_64: u64 = upcast(x8);
    let upcast_x8_to_128: u128 = upcast(x8);

    let upcast_x16_to_16: u16 = upcast(x16);
    let upcast_x16_to_32: u32 = upcast(x16);
    let upcast_x16_to_64: u64 = upcast(x16);
    let upcast_x16_to_128: u128 = upcast(x16);

    let upcast_x32_to_32: u32 = upcast(x32);
    let upcast_x32_to_64: u64 = upcast(x32);
    let upcast_x32_to_128: u128 = upcast(x32);

    let upcast_x64_to_64: u64 = upcast(x64);
    let upcast_x64_to_128: u128 = upcast(x64);

    let upcast_x128_to_128: u128 = upcast(x128);

    // Downcasts.
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
}
