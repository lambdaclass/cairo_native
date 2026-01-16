use core::pedersen;

fn main() -> (felt252, felt252, felt252) {
    (
        pedersen::pedersen(1, 2),
        pedersen::pedersen(3, 4),
        pedersen::pedersen(5, 6),
    )
}
