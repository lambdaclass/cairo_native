use core::array::{tuple_from_span, FixedSizedArrayInfoImpl};

fn run_test(x: Array<felt252>) -> Option<@Box<[core::felt252; 3]>> {
    tuple_from_span::<[felt252; 3], FixedSizedArrayInfoImpl<felt252, 3>>(@x)
}