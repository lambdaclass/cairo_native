fn init_dict(length: u64) -> Felt252Dict<felt252> {
    let mut balances: Felt252Dict<felt252> = Default::default();

    for i in 0..length {
        let x: felt252 = i.into();
        balances.insert(x, x);
    };

    return balances;
}

fn main() {
    let mut dict = init_dict(1001);
    let last = dict.get(1000);
    assert(
        last == 1000,
        'invalid result'
    );
}
