use debug::PrintTrait;
use hash::pedersen;

fn test_pedersen() -> felt252 {
    pedersen(pedersen(pedersen(1, 2), 3), 4)
}

fn main() {
    test_pedersen().print();
}
