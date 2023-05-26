// from starkware.cairo.common.bool import TRUE, FALSE
// from starkware.cairo.common.alloc import alloc

// func compare_arrays(array_a: felt*, array_b: felt*, array_length: felt, iterator: felt) -> (
//     r: felt
// ) {
//     if (iterator == array_length) {
//         return (TRUE,);
//     }
//     if (array_a[iterator] != array_b[iterator]) {
//         return (FALSE,);
//     }
//     return compare_arrays(array_a, array_b, array_length, iterator + 1);
// }

// func fill_array(array: felt*, base: felt, step: felt, array_length: felt, iterator: felt) {
//     if (iterator == array_length) {
//         return ();
//     }
//     assert array[iterator] = base + step * iterator;
//     return fill_array(array, base, step, array_length, iterator + 1);
// }

// func main() {
//     alloc_locals;
//     tempvar array_length = 250000;
//     let (array_a: felt*) = alloc();
//     let (array_b: felt*) = alloc();
//     fill_array(array_a, 7, 3, array_length, 0);
//     fill_array(array_b, 7, 3, array_length, 0);
//     let result: felt = compare_arrays(array_a, array_b, array_length, 0);
//     assert result = TRUE;
//     return ();
// }

use array::ArrayTrait;
use traits::Into;

fn compare_arrays(
    data_a: Span<felt252>, data_b: Span<felt252>, array_length: u32, iterator: u32
) -> bool {
    if iterator == array_length {
        return true;
    }

    if *data_a[iterator] != *data_b[iterator] {
        return false;
    }

    compare_arrays(data_a, data_b, array_length, iterator + 1)
}

fn fill_array(
    ref data: Array<felt252>, base: felt252, step: felt252, array_length: u32, iterator: u32
) {
    if iterator == array_length {
        return ();
    }

    data.append(base + step * iterator.into());
    fill_array(ref data, base, step, array_length, iterator + 1);
}

fn main() {
    let array_length = 200000;

    let mut array_a = ArrayTrait::<felt252>::new();
    let mut array_b = ArrayTrait::<felt252>::new();

    fill_array(ref array_a, 7, 3, array_length, 0);
    fill_array(ref array_b, 7, 3, array_length, 0);

    let result = compare_arrays(array_a.span(), array_b.span(), array_length, 0);
    assert(result == true, 'Arrays are not equal');
}
