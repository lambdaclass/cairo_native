fn mul_if_not_zero(a: felt) -> felt {
    match a {
        0 => 0,
        _ => a * 2,
    }
}
