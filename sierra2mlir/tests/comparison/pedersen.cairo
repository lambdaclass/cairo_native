use hash::pedersen;

fn main() -> (felt252, felt252, felt252) {
    (
        pedersen(1, 2),
        pedersen(3, 4),
        pedersen(5, 6),
    )
}
