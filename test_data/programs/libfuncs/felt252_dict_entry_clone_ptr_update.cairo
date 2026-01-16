use core::dict::Felt252Dict;

fn run_test() {
    let mut dict: Felt252Dict<u64> = Default::default();

    let snapshot = @dict;
    dict.insert(1, 1);
    drop(snapshot);

    dict.insert(2, 2);
}