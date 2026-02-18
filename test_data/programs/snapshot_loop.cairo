use core::dict::Felt252Dict;

fn run_test() {
    let mut dict: Felt252Dict<u64> = Default::default();

    for number in 0..50_u64 {
        let snapshot = @dict;

        let key = number.try_into().unwrap();
        dict.insert(key, number);

        drop(snapshot)
    }
}
#[inline(never)]
// Force T to stay alive until after the end of the scope.
fn drop<T,+Drop<T>>(_x: T) {
}
