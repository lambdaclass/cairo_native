use core::dict::{Felt252Dict, Felt252DictEntryTrait, SquashedFelt252DictImpl};

pub fn main() {
    // The squash libfunc has a fixed range check cost of 2.

    // If no big keys, 3 per unique key access.
    let mut dict: Felt252Dict<felt252> = Default::default();
    dict.insert(1, 1); // 3
    dict.insert(2, 2); // 3
    dict.insert(3, 3); // 3
    dict.insert(4, 4); // 3
    dict.insert(5, 4); // 3
    dict.insert(6, 4); // 3
    let _ = dict.squash(); // 2
    // SUBTOTAL: 20

    // A dictionary has big keys if there is at least one key greater than
    // the range check bound (2**128 - 1).

    // If has big keys, 2 for first unique key access,
    // and 6 each of the remaining unique key accesses.
    let mut dict: Felt252Dict<felt252> = Default::default();
    dict.insert(1, 1); // 2
    dict.insert(0xF00000000000000000000000000000002, 1); // 6
    dict.insert(3, 1); // 6
    dict.insert(0xF00000000000000000000000000000004, 1); // 6
    dict.insert(5, 1); // 6
    dict.insert(0xF00000000000000000000000000000006, 1); // 6
    dict.insert(7, 1); // 6
    let _ = dict.squash(); // 2
    // SUBTOTAL: 40


    // If no big keys, 3 per unique key access.
    // Each repeated key adds an extra range check usage.
    let mut dict: Felt252Dict<felt252> = Default::default();
    dict.insert(1, 1); // 3
    dict.insert(2, 1); // 3
    dict.insert(3, 1); // 3
    dict.insert(4, 1); // 3
    dict.insert(1, 1); // 1
    dict.insert(2, 1); // 1
    dict.insert(1, 1); // 1
    dict.insert(2, 1); // 1
    dict.insert(1, 1); // 1
    dict.insert(2, 1); // 1
    let _ = dict.squash(); // 2
    // SUBTOTAL: 20


    // If has big keys, 2 for first unique key access,
    // and 6 each of the remaining unique key accesses.
    // Each repeated key access adds an extra range check usage.
    let mut dict: Felt252Dict<felt252> = Default::default();
    dict.insert(1, 1); // 2
    dict.insert(0xF00000000000000000000000000000002, 1); // 6
    dict.insert(1, 1); // 1
    dict.insert(0xF00000000000000000000000000000002, 1); // 1
    dict.insert(1, 1); // 1
    dict.insert(0xF00000000000000000000000000000002, 1); // 1
    dict.insert(1, 1); // 1
    dict.insert(0xF00000000000000000000000000000002, 1); // 1
    dict.insert(1, 1); // 1
    dict.insert(0xF00000000000000000000000000000002, 1); // 1
    dict.insert(1, 1); // 1
    dict.insert(0xF00000000000000000000000000000002, 1); // 1
    let _ = dict.squash(); // 2
    // SUBTOTAL: 20

    // TOTAL: 100
}
