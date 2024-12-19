fn init_dict(length: u64) -> Felt252Dict<felt252> {
    let mut balances: Felt252Dict<felt252> = Default::default();

    for i in 0..length {
        let x: felt252 = i.into();
        balances.insert(x, x);
    };

    return balances;
}

#[derive(Destruct)]
struct VM {
    marker: u64,
    dict: Felt252Dict<felt252>,
}

fn main() {
    let dict = init_dict(1000);
    let vm = VM {
        dict: dict,
        marker: 1,
    };

    let dups: u64 = 10000;
    let mut counter: u64 = 0;

    for _ in 0..dups {
        let vmsnap = @vm;
        let vmmarker = vmsnap.marker;
        counter += *vmmarker;
    };

    assert(
        counter == dups,
        'invalid result'
    );
}
