fn main() -> (
    (u8, u8, bool, bool),
    (u16, u16, bool, bool),
    (u32, u32, bool, bool),
    (u64, u64, bool, bool),
    (u128, u128, bool, bool)
) {
    (
        (min_u8(2_u8, 4_u8), min_eq_u8(2_u8, 5_u8), eq_u8(2_u8, 2_u8), eq_u8(2_u8, 3_u8)),
        (min_u16(2_u16, 4_u16), min_eq_u16(2_u16, 6_u16), eq_u16(2_u16, 2_u16), eq_u16(2_u16, 3_u16)),
        (min_u32(2_u32, 4_u32), min_eq_u32(2_u32, 7_u32), eq_u32(2_u32, 2_u32), eq_u32(2_u32, 3_u32)),
        (min_u64(2_u64, 4_u64), min_eq_u64(2_u64, 8_u64), eq_u64(2_u64, 2_u64), eq_u64(2_u64, 3_u64)),
        (min_u128(2_u128, 4_u128), min_eq_u128(2_u128, 9_u128), eq_u128(2_u128, 2_u128), eq_u128(2_u128, 3_u128)),
    )
}

fn min_u8(a: u8, b: u8) -> u8 {
    if a < b {
        a
    } else {
        b
    }
}

fn min_eq_u8(a: u8, b: u8) -> u8 {
    if a <= b {
        a
    } else {
        b
    }
}

fn eq_u8(a: u8, b: u8) -> bool {
    if a == b {
        true
    } else {
        false
    }
}


fn min_u16(a: u16, b: u16) -> u16 {
    if a < b {
        a
    } else {
        b
    }
}

fn min_eq_u16(a: u16, b: u16) -> u16 {
    if a <= b {
        a
    } else {
        b
    }
}

fn eq_u16(a: u16, b: u16) -> bool {
    if a == b {
        true
    } else {
        false
    }
}

fn min_u32(a: u32, b: u32) -> u32 {
    if a < b {
        a
    } else {
        b
    }
}

fn min_eq_u32(a: u32, b: u32) -> u32 {
    if a <= b {
        a
    } else {
        b
    }
}

fn eq_u32(a: u32, b: u32) -> bool {
    if a == b {
        true
    } else {
        false
    }
}

fn min_u64(a: u64, b: u64) -> u64 {
    if a < b {
        a
    } else {
        b
    }
}

fn min_eq_u64(a: u64, b: u64) -> u64 {
    if a <= b {
        a
    } else {
        b
    }
}

fn eq_u64(a: u64, b: u64) -> bool {
    if a == b {
        true
    } else {
        false
    }
}

fn min_u128(a: u128, b: u128) -> u128 {
    if a < b {
        a
    } else {
        b
    }
}

fn min_eq_u128(a: u128, b: u128) -> u128 {
    if a <= b {
        a
    } else {
        b
    }
}

fn eq_u128(a: u128, b: u128) -> bool {
    if a == b {
        true
    } else {
        false
    }
}
