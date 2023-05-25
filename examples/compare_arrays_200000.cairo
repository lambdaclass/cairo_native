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

fn compare_arrays(array_a: Array<felt252>, array_b: Array<felt252>, array_length: felt252, iterator: felt252) -> bool {
    if (iterator == array_length) {
        return true;
    }
    if (array_a[iterator] != array_b[iterator]) {
        return false;
    }
    return compare_arrays(array_a, array_b, array_length, iterator + 1);
}

fn fill_array(array: Array<felt252>, base: felt252, step: felt252, array_length: felt252, iterator: felt252) {
    if (iterator == array_length) {
        return ();
    }
    let array[iterator] = base + step * iterator;
    return fill_array(array, base, step, array_length, iterator + 1);
}

#[test]
fn main() {
    let array_length: felt252 = 250000;
    let array_a: Array<felt252> = ArrayTrait::new();
    let array_b: Array<felt252> = ArrayTrait::new();
    fill_array(array_a, 7, 3, array_length, 0);
    fill_array(array_b, 7, 3, array_length, 0);
    let result: felt252 = compare_arrays(array_a, array_b, array_length, 0);
    assert(result == true, 'test failed');
    return ();
}